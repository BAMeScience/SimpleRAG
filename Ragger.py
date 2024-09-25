from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
import bs4
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
import ollama
from langchain.embeddings import HuggingFaceBgeEmbeddings
import json
import faiss
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryByteStore
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import uuid
from langchain_experimental.text_splitter import SemanticChunker

from langchain_core.prompts import ChatPromptTemplate
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
#### for later improvements: https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb?ref=blog.langchain.dev
#### https://python.langchain.com/v0.2/docs/tutorials/rag/
import ollama

class RAGRetriever():
    def __init__(self,text_path,device='cuda',embed_device='cpu',vecStoreDir = './',GeneratorModel='llama3:70b', temp=0):
        self.text_txt = open(text_path, "r").read()
        self.text = TextLoader(text_path).load()
        '''
        self.embedding_function = OllamaEmbeddings(
            model="llama3:8b", temperature=0)        
        '''          
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': embed_device}
        encode_kwargs = {"normalize_embeddings": True}

        self.embedding_function = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)

        self.vecStoreDir = vecStoreDir
        self.chatModel = ChatOllama(model=GeneratorModel, temperature=temp, 
                                    device=device)
    
    
    def update_chat_model(self, GeneratorModel='llama3:70b', temp=0, device='cuda'):
        self.chatModel = ChatOllama(model=GeneratorModel, temperature=temp, device=device)

    def get_chunks(self,chunk_size =1000,chunk_overlap=500):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            #separators=["\n\n"], 
            length_function=len,
            is_separator_regex=False,
        )

        #texts = text_splitter.create_documents([self.text])
        texts = text_splitter.split_documents(self.text)
        
        '''
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size*.2,
            chunk_overlap=chunk_overlap*.2,
            length_function=len,
            is_separator_regex=False,
        )
        texts2 = text_splitter.split_documents(self.text)

        texts = texts + texts2
        '''
        self.chunks = texts
        return texts
    
    def get_chunks_2(self, chunk_size, chunk_overlap):
        chunks = []
        start = 0
        while start < len(self.text):
            end = min(start + chunk_size, len(self.text))
            chunks.append(self.text[start:end])
            # Move the start to the next chunk, ensuring overlap
            start += chunk_size - chunk_overlap
        return chunks
    def semantic_split(self,docs: list[Document]) -> list[Document]:
        """
        Semantic chunking

        Args:
            docs (List[Document]): List of documents to chunk

        Returns:
            List[Document]: List of chunked documents
        """
        splitter = SemanticChunker(
            self.embedding_function, breakpoint_threshold_type="gradient"
        )
        return splitter.split_documents(docs)
    def createVecStore_multiVec(self,chunk_size =1000,chunk_overlap=500):
        store = InMemoryByteStore()
        id_key = "doc_id"
        # The retriever (empty to start)
        docs = self.get_chunks(chunk_size =chunk_size,chunk_overlap=chunk_overlap)
        
        
        retriever = MultiVectorRetriever(
            vectorstore=Chroma(collection_name="summaries", embedding_function=self.embedding_function),
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": 4},
            search_type='mmr')
        '''
        ###### Faiss#######
        self.index = faiss.IndexFlatL2(len(self.embedding_function.embed_query("hello world")))
        vectorstore = FAISS.from_documents(
        documents=docs,             # Your list of documents
        docstore=InMemoryDocstore(),
        embedding=self.embedding_function,         # Your embedding function
        distance_strategy="MAX_INNER_PRODUCT" # Use dot product similarity
        )

        retriever = MultiVectorRetriever(
            vectorstores=vectorstore,  # You can add more vectorstores here
            search_kwargs={"k": 2},
            byte_store=store,
            id_key=id_key,      # Number of nearest neighbors to retrieve
        )
        ##### Faiss#######
        '''
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                                             chunk_overlap=50,
                                                             length_function=len,
                                                             is_separator_regex=False)
        sub_docs = []
        for i, doc in enumerate(docs):
            _id = doc_ids[i]
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata[id_key] = _id
                #_doc.metadata["original_chunk"] = doc.page_content  # Store original chunk content

            sub_docs.extend(_sub_docs)
        
        
        retriever.vectorstore.add_documents(sub_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))
        #retriever1 = MultiQueryRetriever.from_llm( retriever=retriever,
        #                                    llm=self.chatModel)
        
        self.MultiRetriever = retriever
        #return retriever, doc_ids

    def get_chunksLLM(self):
        prompt= """
                For processing text it is essential to chunk text for LLMs.
                Chunks must be coherrent and related.
                You are an expert on chunking and therefore you will be passed a text to chunk.
                Instructions:
                1- Related text must be in one chunk, just add the seperator "chunk_here"
                before each chunk. 
                2- You must not to modify or delete and text and must reproduce the whole text.
                3- Read each line and understand the relation to the previous text.
                Adhere strictly to the instructions.
                Here is the text:
                """
        
        prompt = prompt + self.text_txt
        response = ollama.chat(model='llama3:8b', messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                    ])
        res = response['message']['content']
        with open("DocLLM.txt", "w") as text_file:
            text_file.write(res)
        print(res)
        return res

    def createVecStore(self,chunk_size =1000,chunk_overlap=500):
        self.db = Chroma.from_documents(self.get_chunks(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                                        , self.embedding_function)#, persist_directory=self.vecStoreDir)
        return self.db

    
    def retrieve(self,Q='query'):
        try:
            retriever = self.db.as_retriever(search_kwargs={"k": 2})
            #retriever1 = self.multiVec()
            #retriever = MultiQueryRetriever.from_llm( retriever=retriever,
            #                                          llm=self.chatModel)
            
            #query_result = retriever.retriever.vectorstore.search(Q)  # Assuming a search method that returns results
            #document_ids_used = [doc.metadata['doc_id'] for doc in query_result]
            self.retrieverTest = retriever    
        except:
            print('Create or load the vector store before,\
                   use the functions: createVecStore() or\
                   load_vectorStore()')
        
        self.prompt = hub.pull("rlm/rag-prompt")


        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.chatModel
            | StrOutputParser()
        )

        return rag_chain.invoke(Q)
    
    def mRetriever(self,Q='query',hint=''):
        #retriever_context, ids = self.multiVec(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        RETRIEVER = self.MultiRetriever 
        #self.used_documents = retriever_context.get_used_documents()
        #CH.MultiRetriever.vectorstore.get()
        # Prompt template
        template = """Answer the question based only on the following context, which can include text and tables. 
        {context}
        Question: {question}
        You are a only a retriever and you should only output the text as requested (copy paste). If the asnwer not found only write 'not found'.
        """
        if len(hint)>0:
            template = template + 'Hint: ' + hint
        prompt = ChatPromptTemplate.from_template(template)
        self.prompt = prompt
        self.template = template
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = {"context": RETRIEVER | format_docs,
                  "question": RunnablePassthrough()} | prompt | self.chatModel | StrOutputParser()
        return chain.invoke(Q)



    



'''
import numpy as np
Q_e = np.array(CH.embedding_function.embed_query(query))
q_lsit = []
for chun in CH.chunks:
    q_lsit.append(CH.embedding_function.embed_query(chun.page_content))
Q_array= np.array(q_lsit)
'''
#CH.retrieverTest.get_relevant_documents(query)






#CH = Chunker1('Doc.txt',device='cuda:0')



#CH.get_chunksLLM()

#CH.db.similarity_search('On which date was the material accepted as BAM-CRM?')
#CH.db.similarity_search('when was it accepted as BAM-CRM?')



#CH.test.similarity_search('what is the storage information?')