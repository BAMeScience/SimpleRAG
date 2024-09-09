import ollama
from ollama import Client
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LLM():
    def __init__(self, model='llama3:8b',device =0):
        self.client = Client(host='http://localhost:11434',)
        self.instructions = open('instructions.txt', "r").read()
        self.segmentTxt = open('segment.txt', "r").read()

        self.model = model
        self.device=device
    def segmenter(self,context):
        prompt_full = f"""{self.segmentTxt} \nContext: {context} \nSegmented Context:"""

        response = self.client.chat(model=self.model, messages=[
        {
            'role': 'user',
            'content': prompt_full,
        }],
        options={"temperature":0, 'main_gpu':self.device})

        #print(response['message']['content'])
        return response['message']['content']
    def getAnswer(self,context,question,temp=0):
        prompt_full = f"""
        {self.instructions}
        \nContext: {context}
        \nQuestion: {question}
        \nAnswer:"""

        response = self.client.chat(model=self.model, messages=[
        {
            'role': 'user',
            'content': prompt_full,
        }],
        options={"temperature":temp, 'main_gpu':self.device})

        #print(response['message']['content'])
        return response['message']['content']   
    def custom(self,context,prompt,temp=.7):
        prompt_full = f"""
        \nContext: {context}
        \n Request:{prompt}
        \nQ&A:"""

        response = self.client.chat(model=self.model, messages=[
        {
            'role': 'user',
            'content': prompt_full,
        }],
        options={"temperature":temp, 'main_gpu':self.device})

        #print(response['message']['content'])
        return response['message']['content']  




#context = open('../Doc3.txt', "r").read()
context = open('/home/balbakri/ragger/Doc3.txt', "r").read()
questions = open('questions_comprehensive.txt', "r").read()
SQuestion = open('questionSpecific.txt', "r").read()
#llm1 = LLM(model='phi3:14b')
#res =llm1.getAnswer(context=context,question=questions)
#outPath = 'responseAtOnce.txt'
#with open(outPath, "w") as text_file:
#    text_file.write(res)



Dox = context.split('new page')
#questions_split = questions.split('\n')[1:]
questions_split = questions.split('[question]')[1:]

intitalAnswers = {}

for k, page in enumerate(Dox):
    res_list = []  
    for i,q in enumerate(questions_split):
        llm1 = LLM(model='llama3:8b')
        res =llm1.getAnswer(context=page,question=q)
        res_list.append(res)
    intitalAnswers[str(k)] = res_list


filteredAnswers = []
FilteredAnswerDict = {}
for iq,q in enumerate(questions_split):
    filteringPrompt = f"""you should solve a multiple choice problem given a context. 
    Entities in the provided context can be slightly different from the answers but semantically the same.
    Here is the provided context: {context}
    Question: What is the correct asnwer from the following choices given the context above: """
    for i in intitalAnswers.keys():
        if '**' in intitalAnswers[i][iq]:
            filteringPrompt = filteringPrompt + 'Answer ['+ str(i) +'] '+ intitalAnswers[i][iq] + '\n'
        else:
            pass
    filteringPrompt = filteringPrompt + 'Only output the number of the correct choice, never write any addtionaly text. Only the number!'
    llm_filtering = LLM(model='llama3:8b')
    res =llm_filtering.custom(context=context,
                              prompt=filteringPrompt,temp=0)
    print(res)
    filteredAnswers.append(res)
    FilteredAnswerDict[q] = intitalAnswers[res][iq]
FilteredAnswerDict[questions_split[0]] = intitalAnswers['0'][0] # title always in the first page



############################# Tables ######################
for D in Dox:

    model = 'llama3:70b' #'llama3:70b' # 'mistral' #
    #task = """given the above text, there are many differnet measurements, different measurement mean independent tables or sets of measurements. how many sets are there?"""
    #question = "units must be accurate &Multiple! Extract all tables in the given context."#"How many set of measurements are in the certificate? a set can be represented in a table. one table of measurements would be set." #
    question = "Extract all tables in the given context with title and footnote if available."#"How many set of measurements are in the certificate? a set can be represented in a table. one table of measurements would be set." #
    prompt_full = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    \nQuestion: {question}"""

    TabLLM = LLM(model='llama3:70b')
    res =TabLLM.custom(context=D,
                              prompt=prompt_full,temp=0)

    print(res)
    print('################################################################')


####################### Shuffle and Temperature ##############



import random
def shuffler(context):

    context_list = context.split('\n')
    random.shuffle(context_list)
    context_new= ''
    for i in context_list:
        context_new = context_new + i
    return context_new

res_list = []  
emb_list= []
q_id = 6
#emb = ollama.embeddings(model='nomic-embed-text:latest',prompt=questions_split[q_id])
#emb_list.append(emb['embedding'])

for i in range(10):
    llm1 = LLM(model='llama3:8b')
    res =llm1.getAnswer(context=shuffler(context),question=questions_split[q_id],temp=.5)
    res_list.append(res)
    emb = ollama.embeddings(model='nomic-embed-text:latest',prompt=res_list[i])
    emb_list.append(emb['embedding'])

emb_array = np.array(emb_list)

for res_i in res_list:
    print(res_i)
    print('#############################################')






##################### Question Generator ########################

def cut_context(context,length=1000):
    st_id = np.random.randint(0,len(context)-length)
    end_id = st_id + length

    return context[st_id:end_id]

llm1 = LLM(model='llama3:8b')

LLM_Questions = []
LLM_Answers = []
for i in range(100):
    res =llm1.custom(context=cut_context(context),
                    prompt="""You will be given text from a document about certified reference materials.
                    your job is to generate a question from the given context and answer it. The answer should be copied as it is exactly written in context wihtout changes.
                    use special characters Q_start and Q_end to indicte start of the question and closing it.
                    the same for the answer with A_start and A_end.
                    for example:
                        Q_start 'What is the period of validity?' Q_end
                        A_start '12 months beginning with the dispatch.' A_end"""
                    ,temp=1)
    
    try: 
        Q_id_start = res.find('Q_start') + len('Q_start')
        Q_id_end = res.find('Q_end') + len('Q_end')

        A_id_start = res.find('A_start') + len('A_start')
        A_id_end = res.find('A_end') + len('A_end')
        LLM_question = res[Q_id_start:Q_id_end]
        LLM_answer = res[A_id_start:A_id_end]
        LLM_Questions.append(LLM_question)
        LLM_Answers.append(LLM_answer)

    except:
        pass




Q_embeddings=  []
for Qu in LLM_Questions:
    emb_Q = ollama.embeddings(model='nomic-embed-text:latest',prompt=Qu)
    Q_embeddings.append(emb_Q['embedding'])
Q_embeddings = np.array(Q_embeddings)

A_embeddings=  []
for Au in LLM_Answers:
    emb_A = ollama.embeddings(model='nomic-embed-text:latest',prompt=Au)
    A_embeddings.append(emb_A['embedding'])
A_embeddings = np.array(A_embeddings)
np.mean(Q_embeddings,A_embeddings).shape



#res =llm1.custom(context=context,prompt="""
#given the context, what is more correct: '**Intended use of the material**: Before withdrawing a sub-sample, the bottle should be allowed to reach room temperature. Thereafter, the bottle should be closed tightly and stored at (4 ± 2) °C. The stability of the reference material is not affected by short periods of handling at ambient temperature during transport and use.' OR '**Intended use of the material**: The intended purpose of CRM BAM -U117 is the verification of analytical results obtained for the mass fraction of total cyanide in soils and soil-like materials applying the standardized procedures DIN ISO 11262:2012 [1] and DIN EN ISO 17380:2013 [2]. As any reference material, it can also be used for routine performance checks (quality control charts) or validation studies.' """)
