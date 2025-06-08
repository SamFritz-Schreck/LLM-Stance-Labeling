'''
packages to install for WIRE environment:
pip install transformers langchain accelerate bitsandbytes einops
'''

import os, re, pandas as pd, numpy as np, ast, json, string, logging, sys, warnings, time
from pprint import pprint
from tqdm import tqdm

'''
Specify Dataset, Model, and Number of Runs. 
Datasets include: covid-lies, srq, wtwt, semeval, election, phemerumours
Decoder only models include: mistralai/Mistral-7B-Instruct-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1, 
microsoft/phi-2, 
meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-13b-chat-hf, meta-llama/Llama-2-70b-chat-hf, 
tiiuae/falcon-7b-instruct, tiiuae/falcon-40b-instruct
facebook/opt-2.7b, facebook/opt-6.7b
Encoder-decoder models include: declare-lab/flan-alpaca-gpt4-xl, declare-lab/flan-alpaca-xxl, google/flan-ul2
'''
#DATASET = "srq"
#MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
DATASET = sys.argv[1]
MODEL = sys.argv[2]
NUM_RUNS = 1
with open('/home/jovyan/huggingface_token.txt', 'r') as file:  
    token = file.read().strip()

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset")

'''
Import DL packages
'''
import torch
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import SequentialChain, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import accelerate

from sklearn.metrics import classification_report

# Create a logger and set level to INFO.  
logger = logging.getLogger()  
logger.setLevel(logging.INFO)  
# Create a file handler for output file.  
handler = logging.FileHandler('labeling_logs/'+DATASET+'_'+MODEL.split("/")[1]+'.log') 
# Add the file handler to logger  
logger.addHandler(handler) 

'''
Helper Functions for Post-processesing, Model creation, and Prompt Generation
'''

def post_process_results(results):
    """
    This function post-processes the results from a large language model to label text.

    Args:
        results (list): A list of strings, each string is a word from language model output.

    Returns:
        list: A list of strings, each string is a classification label ('disagree', 'neutral', 'agree').
    """

    # A list to store the prediction labels
    y_pred = []  

    # Words or phrases that indicate each stance category
    disagree_indicators = ['against', 'denies', 'critical', 'deny', 'neg', 'oppose', 'opposes']
    agree_indicators = ['support','supports', 'for', 'pro ', 'positive', 'agree', 'agrees']
    neutral_indicators = ['neutral', 'unrelated']

    # Iterate over LLM return in the results
    for word in results:  
        # Normalize the word to lower case and remove leading/trailing white spaces
        normalized_word = str(word).strip().lower()

        if any(indicator in normalized_word for indicator in disagree_indicators):
            # If the word is also found in agree_indicators or neutral_indicators, label it as 'neutral'
            if any(indicator in normalized_word for indicator in agree_indicators) or any(indicator in normalized_word for indicator in neutral_indicators):
                y_pred.append('neutral')
            else:
                y_pred.append('disagree')
        elif any(indicator in normalized_word for indicator in neutral_indicators):
            # If the word is also found in disagree_indicators or agree_indicators, label it as 'neutral'
            if any(indicator in normalized_word for indicator in disagree_indicators) or any(indicator in normalized_word for indicator in agree_indicators):
                y_pred.append('neutral')
            else:
                y_pred.append('neutral')
        elif any(indicator in normalized_word for indicator in agree_indicators):
            # If the word is also found in disagree_indicators or neutral_indicators, label it as 'neutral'
            if any(indicator in normalized_word for indicator in disagree_indicators) or any(indicator in normalized_word for indicator in neutral_indicators):
                y_pred.append('neutral')
            else:
                y_pred.append('agree')
        else:  
            # If no specific stance label is found, label it as 'neutral'
            y_pred.append('neutral')


    return y_pred

def create_task_chain(llm):
    # task-only prompt
    
    #task_template = '''Classify the following statement as to whether it is for, against, or neutral. Only return the stance for the statement, and no other text.
    #statement: {statement}
    #stance:'''

    #task_template = '''What is the stance of the following statement? Give the stance as for, against, neutral, or unrelated. Only return the stance and no other text.
    #statement: {statement}
    #stance:'''

    task_template = '''Analyze the following statement and determine its stance. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
    statement: {statement}
    stance:'''
    

    task_prompt = PromptTemplate(
        input_variables=["statement"],
        template=task_template
    )

    return LLMChain(prompt=task_prompt, llm=llm)

def create_definition_chain(llm):
    # definition and task prompt

    #definition_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Classify the stance of the following statement as to whether it is for, against, or neutral. Only return the stance for the statement, and no other text.
    #statement: {statement}
    #stance:'''

    #definition_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following statement? Give the stance as for, against, neutral, or unrelated. Only return the stance and no other text.
    #statement: {statement}
    #stance:'''

    definition_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following statement and determine its stance. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
    statement: {statement}
    stance:'''

    definition_prompt = PromptTemplate(
        input_variables=["statement"],
        template=definition_template
    )

    return LLMChain(prompt=definition_prompt, llm=llm)

def create_context_analyze_chain(dataset, llm):
    # context prompt
    #Classify the stance of the following reply as to whether it supports, denies, or neutral toward the statement.
    #Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media satement toward the following entity? Give the stance as for, against, neutral, or unrelated. Only return the stance and no other text.

    if dataset == 'semeval':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        entity: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset == 'election':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided politician. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        politician: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset == 'srq':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following reply and determine its stance toward the provided social media statement about the provided event. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        event: {event}
        statement: {statement}
        reply: {reply}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement", "reply"],
            template=context_template
        )
    elif dataset == 'phemerumours':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided rumor. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        rumor: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset =='covid-lies':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided belief about COVID. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        belief: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset =='wtwt':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided merger actually happening. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        merger: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}' was given.")

    return LLMChain(prompt=context_prompt, llm=llm)

def create_context_question_chain(dataset, llm):
    # context prompt
    #Classify the stance of the following reply as to whether it supports, denies, or neutral toward the statement.
    #Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media statement toward the following entity? Give the stance as for, against, neutral, or unrelated. Only return the stance and no other text.

    if dataset == 'semeval':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media statement toward the following entity? Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        entity: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset == 'election':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media statement toward the following politician? Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        politician: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset == 'srq':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following reply toward the following social media statement about the following event? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        event: {event}
        statement: {statement}
        reply: {reply}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement", "reply"],
            template=context_template
        )
    elif dataset == 'phemerumours':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media statement toward the following rumor? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        rumor: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset =='covid-lies':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media statement toward the following belief about COVID? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        belief: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset =='wtwt':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media satement toward the following merger actually happening? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        merger: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}' was given.")

    return LLMChain(prompt=context_prompt, llm=llm)

def create_few_shot_chain(dataset, llm):
    # few-shot prompt
    #Now, classify the stance of the following statement toward the following entity.
    #What is the stance of the following social media satement toward the following event? Give the stance as for, against, neutral, or unrelated. Only return the stance and no other text.

    if dataset == "semeval":
        example_template = '''entity: {entity}
        statement: {statement}
        stance: {stance}'''

        example_prompt = PromptTemplate(
            input_variables=["entity","statement", "stance"],
            template=example_template
        )

        examples = [
            {'entity': "Atheism",
            'statement':"Leaving Christianity enables you to love the people you once rejected. #freethinker #Christianity #SemST",
            'stance': 'for'},
            {'entity': "Climate Change is a Real Concern",
            'statement':"@AlharbiF I'll bomb anything I can get my hands on, especially if THEY aren't christian. #graham2016 #GOP #SemST",
            'stance': 'neutral'},
            {'entity': "Feminist Movement",
            'statement':"Always a delight to see chest-drumming alpha males hiss and scuttle backwards up the wall when a feminist enters the room. #manly #SemST",
            'stance': 'for'},
            {'entity': "Hillary Clinton",
            'statement':"Would you wanna be in a long term relationship with some bitch that hides her emails, & lies to your face? Then #Dontvote #SemST",
            'stance': 'against'},
            {'entity': "Legalization of Abortion",
            'statement':"@k_yoder That lady needs help, mental illness is a serious issue. #SemST",
            'stance': 'neutral'},
        ]

        prefix = """Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. The following statements are social media posts expressing opinions about entities. Each statement can either be for, against, neutral, or unrelated toward their associated entity."""

        suffix = '''Analyze the following social media statement and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        entity: {event}
        statement: {statement}
        stance:'''

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["event", "statement"],
            example_separator="\n"
        )

    elif dataset == 'election':
        example_template = '''politician: {politician}
        statement: {statement}
        stance: {stance}'''

        example_prompt = PromptTemplate(
            input_variables=["politician","statement", "stance"],
            template=example_template
        )

        examples = [
            {'politician':"Donald Trump",
            'statement':'''How can you vote for someone just because they are woman, despite #HillaryClinton being a liar? #Trump2016 #trumpforpresident''',
            'stance': 'for'},
            {'politician':"Donald Trump",
            'statement':'''@Samstwitch #HillaryClinton doesn't take questions, but #DonaldTrump just kicks out anyone who doesn't agree with him.''',
            'stance': 'agianst'},
            {'politician':"Donald Trump",
            'statement':'''I want a debate b/t Donald Trump and Hilary Clinton but it's their spouses on stage instead #DonaldTrump #HillaryClinton #debate''',
            'stance': 'neutral'},
            {'politician':"Hilary Clinton",
            'statement':'''It is interesting that the cameras at #Bernie and #Hillary rallies actual move and show crowds Makes your go hmmm #tcot #Trump2016''',
            'stance': 'against'},
            {'politician':"Hilary Clinton",
            'statement':'''#Unions work toward increasing middle class wages w/ collective bargaining &amp; most have endorsed #HillaryClinton not #BernieSanders \n#lastword''',
            'stance': 'for'}
        ]

        prefix = """Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. The following statemements are from social media posts expressing opinions about politicians. Each statement can either be for, against, neutral, or unrelated toward its associated politician."""

        suffix = '''Analyze the following social media statement and determine its stance towards the provided politician. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        politician: {event}
        statement: {statement}
        stance:'''

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["event", "statement"],
            example_separator="\n"
        )

    elif dataset == 'srq':
        example_template = '''event: {event}
        statement: {statement}
        reply: {reply}
        stance: {stance}'''

        example_prompt = PromptTemplate(
            input_variables=["event","statement", "reply", "stance"],
            template=example_template
        )

        examples = [
            {'event': "A school shooting in Santa Fe",
            'statement':"Former GOP Rep. Jason Chaffetz: 'Politically correct culture' to blame for shooting https://t.co/mjagRQZQfr by @EricBoehlert",
            'reply': "Ma calls BS! https://t.co/bodEWN5Q4C",
            'stance': "denies"},
            {'event': "A school shooting in Santa Fe",
            'statement':"LIE! Jimmy Kimmel Pushes Bogus School Shooting Stats That Have BeenDisproven https://t.co/CjefwaKs5S https://t.co/VHK059uuOc",
            'reply': "Why you gotta lie Jimmy? https://t.co/3pkqnWMuw5",
            'stance': "supports"},
            {'event': "The U.S. is leaving the Iran Nuclear Deal",
            'statement':"Iran's president says it will negotiate staying in nuclear deal with European countries Russia and China despite U.S. withdrawal https://t.co/jhMv4VaGl5",
            'reply': "BBC is meanwhile reporting fake news #IranNuclearDeal #Trump https://t.co/J4PlUtLAA3",
            'stance': "neutral"},
            {'event': "The U.S. is leaving the Iran Nuclear Deal",
            'statement':"When you're paid for years to launch disingenuous attacks at the Iran Deal at least own the consequences of blowing it up.. https://t.co/zTt1YFut8O",
            'reply': "You confessed to Obama and admin lying about it to the American people in order to get it done.Sit this one out Ben.The future is not going to work out in your political favor @brhodes. https://t.co/hf6fGJZThF",
            'stance': "denies"},
            {'event': 'Students participating in "March for Our Lives"',
            'statement':"Look at the crowd size for #MarchForOurLives I'd say at least 5X more than at Trump 2017 inauguration!  https://t.co/peJx374CPL",
            'reply': "@POTUS @PressSec @librarycongress I wonder what stupid LIES Sarah Sanders will hurl to start this week off ?? ALIENS invaded D.C. perhaps? https://t.co/HXH7teOJ8e",
            'stance': "supports"}
        ]

        prefix = """Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. The following statements are social media posts about an event with an associated reply. Each reply can either support, deny, be neutral, or be unrelated toward its statement."""

        suffix = '''Analyze the following reply and determine its stance toward the provided social media statement about the provided event. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        event: {event}
        statement: {statement}
        reply: {reply}
        stance:'''

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["event", "statement", "reply"],
            example_separator="\n"
        )

    elif dataset == 'phemerumours':
        example_template = '''rumor: {rumor}
        statement: {statement}
        stance: {stance}'''

        example_prompt = PromptTemplate(
            input_variables=["rumor","statement", "stance"],
            template=example_template
        )

        examples = [
            {'rumor':"Putin has gone missing",
            'statement':"Putin reappears on TV amid claims he is unwell and under threat of coup http://t.co/YZln23EUx1 http://t.co/ZsAnBa5gz3",
            'stance': 'denies'},
            {'rumor':"Michael Essien contracted Ebola",
            'statement': '''What? "@FootballcomEN: Unconfirmed reports claim that Michael Essien has contracted Ebola. http://t.co/GsEizhwaV7"''',
            'stance': 'neutral'},
            {'rumor':"A Germanwings plane crashed",
            'statement': '''@thatjohn @planefinder why would they say urgence in lieu of mayday which is standard ?''',
            'stance': 'neutral'},
            {'rumor':"There is a hostage situation in Sydney",
            'statement': '''@KEEMSTARx dick head it's not confirmed its Jihadist extremists. Don't speculate''',
            'stance': 'neutral'},
            {'rumor':"singer Prince will play a secret show in Toronto",
            'statement': '''OMG. #Prince rumoured to be performing in Toronto today. Exciting!''',
            'stance': 'supports'}
        ]

        prefix = """Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. The following are social media posts commenting on whether a rumor is true. Each statement can either support, deny, be neutral, or be unrelated toward their associated rumor."""

        suffix = '''Analyze the following social media statement and determine its stance towards the provided rumor. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        rumor: {event}
        statement: {statement}
        stance:'''

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["event", "statement"],
            example_separator="\n"
        )

    elif dataset =='covid-lies':
        example_template = '''belief: {belief}
        statement: {statement}
        stance: {stance}'''

        example_prompt = PromptTemplate(
            input_variables=["belief","statement", "stance"],
            template=example_template
        )

        examples = [
            {'belief':"Africans are more resistant to coronavirus.",
            'statement':'''Happen now Blacks are Immune to the coronavirus ' there is a GOD https://t.co/LRq7SZYK0G''',
            'stance': 'supports'},
            {'belief':"Alex Jones' silver-infused toothpaste kills COVID-19",
            'statement':'''#China #COVID-19 As work resumes in outbreak, brand-new 'normal' emerges https://t.co/VENOSSOnx5 https://t.co/RQoeSWoaHH''',
            'stance': 'unrelated'},
            {'belief':"COVID-19 is only as deadly as the seasonal flu.",
            'statement':'''@islandmonk @Stonekettle Closer to 650,000 people will die of the flu this year. The figure of 30,000 is just in the U.S. \nBut the flu has approximately 0.1% mortality vs 2% for COVID-19. Do the math.''',
            'stance': 'denies'},
            {'belief':"Coronavirus is genetically engineered.",
            'statement':'''@TheMadKiwi3 @goodfoodgal nah. A biological warfare agent would kill 99% of its victims, not 2% like the corona virus. This is a naturally occurring virus.''',
            'stance': 'denies'},
            {'belief':"SARS-CoV-2 can survive for weeks on surfaces.",
            'statement':'''Coronavirus could survive up to 9 days outside the body, study says https://t.co/JUzdJgc5Dz''',
            'stance': 'supports'}
        ]

        prefix = """Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. The following statements are social media posts about beleifs about COVID or Coronavirus. The statements can support, deny, be neutral, or be unrelated toward its associated COVID belief."""

        suffix = '''Analyze the following social media statement and determine its stance towards the provided belief about COVID. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        belief: {event}
        statement: {statement}
        stance:'''

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["event", "statement"],
            example_separator="\n"
        )

    elif dataset =='wtwt':
        example_template = '''merger: {event}
        statement: {statement}
        stance: {stance}'''

        example_prompt = PromptTemplate(
            input_variables=["event","statement", "stance"],
            template=example_template
        )

        examples = [
            {'event':"Aetna buying Humana",
            'statement':'''Talk of Aetna, Anthem acquisition moves reaches fever pitch http://t.co/lxjHa7eXWh''',
            'stance': 'unrelated'},
            {'event':"Anthem buying Cigna",
            'statement':'''Sr. Mktg Advisor Aetna may acquire Humana or Cigna http://t.co/lTjwFi0Y8O #marketing #strategy''',
            'stance': 'supports'},
            {'event':"Cigna buying Express Scripts",
            'statement':'''Following the lead of CVS's $CVS acquisition of Aetna $AET and Cigna's $CI acquisition of Express Scripts Holding $ESRX, Walmart $WMT and Humana Inc $HUM are in preliminary talks focusing on possible partnership, or Walmart acquisition of Humana https://t.co/H2FWYwFJYz''',
            'stance': 'unrelated'},
            {'event':"CVS Health buying Aetna",
            'statement':'''@IngrahamAngle @realDonaldTrump He needs to block @cvshealth acquisition of @Aetna, and @Cigna acquisition of @ExpressScripts.  Patients will suffer.''',
            'stance': 'denies'},
            {'event':"Disney buying 21st Century Fox",
            'statement':'''Finally saw #Fant4stic \nImmediately regretted it.''',
            'stance': 'unrelated'},
        ]

        prefix = """Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. The following statements are social media posts that may be commenting on a corporate merger. Each statement can either support, deny, be neutral, or be unrelated toward their merger happening."""

        suffix = '''Analyze the following social media statement and determine its stance towards the provided merger actually happening. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        merger: {event}
        statement: {statement}
        stance:'''

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["event", "statement"],
            example_separator="\n"
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}' was given.")

    return LLMChain(prompt=few_shot_prompt, llm=llm)

def create_zero_shot_CoT_chain(dataset, llm):
    # zero-shot Chain-of-Thought Prompt
    if dataset == "semeval":
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (for, against, neutral, or unrelated) of the social media statement towards the entity.
        entity: {event}
        statement: {statement}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        entity: {event}
        statement: {statement}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")
    
    elif dataset == 'election':
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (for, against, neutral, or unrelated) of the social media statement towards the politician.
        politician: {event}
        statement: {statement}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        politician: {event}
        statement: {statement}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")

    elif dataset == 'srq':
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (supports, denies, neutral or unrelated) of the reply towards the social media statement about the following event.
        event: {event}
        statement: {statement}
        reply: {reply}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement", "reply"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        event: {event}
        statement: {statement}
        reply: {reply}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","reply", "stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")

        llm_chain = SequentialChain(
            chains=[cot_chain_1, cot_chain_2],
            input_variables = ["event", "statement", "reply"],
            output_variables=["label"]
        )

        return llm_chain

    elif dataset == 'phemerumours':
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (support, deny, neutral, or unrelated) of the social media statement towards the rumor.
        rumor: {event}
        statement: {statement}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        rumor: {event}
        statement: {statement}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")

    elif dataset =='covid-lies':
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (support, deny, neutral, or unrelated) of the social media statement towards the belief about COVID.
        belief: {event}
        statement: {statement}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        belief: {event}
        statement: {statement}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")

    elif dataset =='wtwt':
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (support, deny, neutral, or unrelated) of the social media statement towards the merger actually happening.
        belief: {event}
        statement: {statement}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        belief: {event}
        statement: {statement}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")

    else:
        raise ValueError(f"Unknown dataset '{dataset}' was given.")

    llm_chain = SequentialChain(
        chains=[cot_chain_1, cot_chain_2],
        input_variables = ["event", "statement"],
        output_variables=["label"]
    )

    return llm_chain

def create_coda_chain(dataset, llm):
    ### CoDA chain

    if dataset == 'srq':
                # Linguist chain
        linguist_template ='''Accurately and concisely explain the linguistic elements in the statment and its reply, and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. Do nothing else.
        statement: {statement}
        reply: {reply}
        explanation:'''

        linguist_prompt = PromptTemplate(
                input_variables=["statement", "reply"],
                template=linguist_template
            )

        linguist_chain = LLMChain(llm=llm, prompt=linguist_prompt, output_key="linguist_analysis")

        # expert chain
        expert_template ='''Accurately and concisely explain the key elements contained in the following statement and its reply, such as characters, events, parties, religions, etc. Also explain their relationship with {event}. Do nothing else.
        statement: {statement}
        reply: {reply}
        explanation:'''

        expert_prompt = PromptTemplate(
                input_variables=["statement", "event", "reply"],
                template=expert_template
            )

        expert_chain = LLMChain(llm=llm, prompt=expert_prompt, output_key="expert_analysis")

        # user chain
        user_template ='''Analyze the following statement and its reply, focusing on the content, hashtags, Internet slang and colloquialisms, emotional tone, implied meaning, and so on. Do nothing else.
        statement: {statement}
        reply: {reply}
        explanation:'''

        user_prompt = PromptTemplate(
                input_variables=["statement", "event", "reply"],
                template=user_template
            )

        user_chain = LLMChain(llm=llm, prompt=user_prompt, output_key="user_analysis")

        # argument for chain
        for_template ='''You think the attitude behind the following reply in a converation about {event} is in support of the following statement. Folloiwng the statement and reply are explanations of the statement and reply by various personas. Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion.
        statement: {statement}
        reply: {reply}
        linguist explanation: {linguist_analysis}
        expert explanation: {expert_analysis}
        heavy social media user explanation: {user_analysis}
        opinion:'''

        for_prompt = PromptTemplate(
                input_variables=["statement", "reply", "linguist_analysis", "expert_analysis", "user_analysis", "event"],
                template=for_template
            )

        for_chain = LLMChain(llm=llm, prompt=for_prompt, output_key="for_opinion")

        # argument against chain
        against_template ='''You think the attitude behind the following reply in a converation about {event} is against the following statement. Folloiwng the statement and reply are explanations of the statement and reply by various personas. Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion.
        statement: {statement}
        reply: {reply}
        linguist explanation: {linguist_analysis}
        expert explanation: {expert_analysis}
        heavy social media user explanation: {user_analysis}
        opinion:'''

        against_prompt = PromptTemplate(
                input_variables=["statement", "reply", "linguist_analysis", "expert_analysis", "user_analysis", "event"],
                template=against_template
            )

        against_chain = LLMChain(llm=llm, prompt=against_prompt, output_key="against_opinion")

        # Final Judgement Chain
        judgement_template ='''Determine whether the following reply in a conversation about {event} is in favor of, neutral, against, or unrelated to the following statement. 
        statement: {statement}
        reply: {reply}
        Arguments that the attitude is in favor: {for_opinion}
        Arguments that the attitude is against: {against_opinion}
        Choose the stance from "for", "against", "neutral", or "unrelated". Answer with only the option above that is most accurate and nothing else.
        stance:'''

        judgement_prompt = PromptTemplate(
                input_variables=["statement","reply", "for_opinion", "against_opinion", "event"],
                template=judgement_template
            )

        judgement_chain = LLMChain(llm=llm, prompt=judgement_prompt, output_key="label")

        llm_chain = SequentialChain(
            chains=[linguist_chain, expert_chain, user_chain, for_chain, against_chain, judgement_chain],
            input_variables = ["event", "statement", "reply"],
            output_variables=["label"]
        )

    else:
        # Linguist chain
        linguist_template ='''Accurately and concisely explain the linguistic elements in the statment and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. Do nothing else.
        statement: {statement}
        explanation:'''

        linguist_prompt = PromptTemplate(
                input_variables=["statement"],
                template=linguist_template
            )

        linguist_chain = LLMChain(llm=llm, prompt=linguist_prompt, output_key="linguist_analysis")

        # expert chain
        expert_template ='''Accurately and concisely explain the key elements contained in the following statement, such as characters, events, parties, religions, etc. Also explain their relationship with {event}. Do nothing else.
        statement: {statement}
        explanation:'''

        expert_prompt = PromptTemplate(
                input_variables=["statement", "event"],
                template=expert_template
            )

        expert_chain = LLMChain(llm=llm, prompt=expert_prompt, output_key="expert_analysis")

        # user chain
        user_template ='''Analyze the following statement, focusing on the content, hashtags, Internet slang and colloquialisms, emotional tone, implied meaning, and so on. Do nothing else.
        statement: {statement}
        explanation:'''

        user_prompt = PromptTemplate(
                input_variables=["statement", "event"],
                template=user_template
            )

        user_chain = LLMChain(llm=llm, prompt=user_prompt, output_key="user_analysis")

        # argument for chain
        for_template ='''You think the attitude behind the following statement is in support of {event}. Folloiwng the statement are explanations of the statement by various personas. Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion.
        statement: {statement}
        linguist explanation: {linguist_analysis}
        expert explanation: {expert_analysis}
        heavy social media user explanation: {user_analysis}
        opinion:'''

        for_prompt = PromptTemplate(
                input_variables=["statement", "linguist_analysis", "expert_analysis", "user_analysis", "event"],
                template=for_template
            )

        for_chain = LLMChain(llm=llm, prompt=for_prompt, output_key="for_opinion")

        # argument against chain
        against_template ='''You think the attitude behind the following statement is against {event}. Folloiwng the statement are explanations of the statement by various personas. Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion.
        statement: {statement}
        linguist explanation: {linguist_analysis}
        expert explanation: {expert_analysis}
        heavy social media user explanation: {user_analysis}
        opinion:'''

        against_prompt = PromptTemplate(
                input_variables=["statement", "linguist_analysis", "expert_analysis", "user_analysis", "event"],
                template=against_template
            )

        against_chain = LLMChain(llm=llm, prompt=against_prompt, output_key="against_opinion")

        # Final Judgement Chain
        judgement_template ='''Determine whether the following statement is in favor of, neutral, against, or unrelated to {event}. 
        statement: {statement}
        Arguments that the attitude is in favor: {for_opinion}
        Arguments that the attitude is against: {against_opinion}
        Choose the stance from "for", "against", "neutral", or "unrelated". Answer with only the option above that is most accurate and nothing else.
        stance:'''

        judgement_prompt = PromptTemplate(
                input_variables=["statement", "for_opinion", "against_opinion", "event"],
                template=judgement_template
            )

        judgement_chain = LLMChain(llm=llm, prompt=judgement_prompt, output_key="label")

        llm_chain = SequentialChain(
            chains=[linguist_chain, expert_chain, user_chain, for_chain, against_chain, judgement_chain],
            input_variables = ["event", "statement"],
            output_variables=["label"]
        )

    return llm_chain

def create_llm(model, token):
    encoder_decoder_models =['google/flan-ul2', 'declare-lab/flan-alpaca'] 

    if any(model.startswith(base) for base in encoder_decoder_models):
        tokenizer = AutoTokenizer.from_pretrained(model, token=token)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16, #quantizing model
            token = token, #hugginface token for gated models
            device_map="auto",
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=50
        )

        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, token=token)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16, #quantizing model
            token = token, #hugginface token for gated models
            device_map="auto",
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=50
        )

        llm = HuggingFacePipeline(pipeline=pipe)

    return llm

if __name__ == "__main__":
    ### read in dataset 
    # file_path = "/home/jovyan/LLM-Stance-Labeling/"+DATASET+"/data_merged.csv"
    # reading in previous labels to add to them
    file_path = "/home/jovyan/LLM-Stance-Labeling/results/"+DATASET+"_"+MODEL.split("/")[1]+".csv"
    df = pd.read_csv(file_path)

    # Create a sample for all examples, if desired
    #df = df.sample(30)

    logger.info("dataset: {}, number of data points: {}".format(DATASET, len(df)))

    ### create model
    llm = create_llm(MODEL, token)

    ### run test examples and log output
    # What is the stance of the following social media post toward the following entity? Give the stance as either for, against, or neutral. Only return the stance and no other text.
    # What is the stance of the following social media post toward the following entity? Classify the stance as for, against, neutral, or unrelated if the post is not about the entity. Only return the stance, and no other text.
    question = '''Analyze the following social media post and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
    entity: U.S. Army
    post: @vondeveen If the Army wants to actually recruit people, maybe stop breaking people and actually prosecute sexual assualt #nomorewar.
    stance:'''
    logger.info(question+llm(question))

    question = '''Analyze the following social media post and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
    entity: U.S. Army
    post: @artfulask I have never seen a pink-eared duck before. #Army
    stance:'''
    logger.info(question+llm(question))

    question = '''Analyze the following social media post and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
    entity: U.S. Army
    post: I think the @Army helped me become disciplined. I would have surely flunked out of college chasing tail if I didn't get some discipline there. #SFL
    stance:'''
    logger.info(question+llm(question))

    ### for num_runs, for each prompting scheme
    ### run the labeling and log the output counts and a classification report

    # create the chains
    task_chain = create_task_chain(llm)
    definition_chain = create_definition_chain(llm)
    context_analyze_chain = create_context_analyze_chain(DATASET, llm)
    context_question_chain = create_context_question_chain(DATASET, llm)
    few_shot_chain = create_few_shot_chain(DATASET, llm)
    zero_shot_cot_chain = create_zero_shot_CoT_chain(DATASET, llm)
    coda_chain = create_coda_chain(DATASET, llm)

    for run in range(NUM_RUNS):
        start_time = time.time()

        logger.info("------------run number {}--------------".format(run))
        
        ## task prompt
        logger.info("\n\n running task-only prompt")
        results = []
        for index, row in tqdm(df.iterrows()):
            if DATASET == 'srq':
                results.append(task_chain.run(event=row['event'], statement=str(row['target_text']).replace('\n', ' ') , reply=str(row['response_text']).replace('\n', ' ') ))
            else:
                results.append(task_chain.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))
        

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['task_only_preds_raw_'+str(run)] = results
        df['task_only_preds_'+str(run)] = y_pred
        
        ## definition prompt
        logger.info("\n\n running definition + task prompt")
        results = []
        for index, row in tqdm(df.iterrows()):
            if DATASET == 'srq':
                results.append(definition_chain.run(event=row['event'], statement=str(row['target_text']).replace('\n', ' ') , reply=str(row['response_text']).replace('\n', ' ') ))
            else:
                results.append(definition_chain.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['task_and_definition_preds_raw_'+str(run)] = results
        df['task_and_definition_preds_'+str(run)] = y_pred
        
        ## context analyze prompt
        logger.info("\n\n running context analyze prompt")
        results = []
        for index, row in tqdm(df.iterrows()):
            if DATASET == 'srq':
                results.append(context_analyze_chain.run(event=row['event'], statement=str(row['target_text']).replace('\n', ' ') , reply=str(row['response_text']).replace('\n', ' ') ))
            else:
                results.append(context_analyze_chain.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['context_analyze_preds_raw_'+str(run)] = results
        df['context_analyze_preds_'+str(run)] = y_pred
        
        ## context question prompt
        logger.info("\n\n running context question prompt")
        results = []
        for index, row in tqdm(df.iterrows()):
            if DATASET == 'srq':
                results.append(context_question_chain.run(event=row['event'], statement=str(row['target_text']).replace('\n', ' ') , reply=str(row['response_text']).replace('\n', ' ') ))
            else:
                results.append(context_question_chain.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['context_question_preds_raw_'+str(run)] = results
        df['context_question_preds_'+str(run)] = y_pred
        
        ## few-shot prompt
        logger.info("\n\n running few-shot prompt")
        results = []
        for index, row in tqdm(df.iterrows()):
            if DATASET == 'srq':
                results.append(few_shot_chain.run(event=row['event'], statement=str(row['target_text']).replace('\n', ' ') , reply=str(row['response_text']).replace('\n', ' ') ))
            else:
                results.append(few_shot_chain.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['few_shot_preds_raw_'+str(run)] = results
        df['few_shot_preds_'+str(run)] = y_pred
        
        ## zero-shot CoT prompt
        logger.info("\n\n running zero-shot CoT prompt")
        results = []
        for index, row in tqdm(df.iterrows()):
            if DATASET == 'srq':
                results.append(zero_shot_cot_chain.run(event=row['event'], statement=str(row['target_text']).replace('\n', ' ') , reply=str(row['response_text']).replace('\n', ' ') ))
            else:
                results.append(zero_shot_cot_chain.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['zero_shot_cot_preds_raw_'+str(run)] = results
        df['zero_shot_cot_preds_'+str(run)] = y_pred
        
        ## CoDA Prompt
        logger.info("\n\n running CoDA prompt")
        results = []
        for index, row in tqdm(df.iterrows()):
            if DATASET == 'srq':
                results.append(coda_chain.run(event=row['event'], statement=str(row['target_text']).replace('\n', ' ') , reply=str(row['response_text']).replace('\n', ' ') ))
            else:
                results.append(coda_chain.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['coda_preds_raw_'+str(run)] = results
        df['coda_preds_'+str(run)] = y_pred
        

        elapsed_time = time.time() - start_time  
        logger.info("Time elapsed for this run: {} seconds".format(elapsed_time))

    df.to_csv("results/"+DATASET+"_"+MODEL.split("/")[1]+".csv", index=False)
