from Ragger import RAGRetriever
from miniCPM import Extract

vector_store_created = False
methods = ['ocr','text']
method = methods[1]


documents_path = "./documents/"
pdf_path = documents_path + "NIST 1849b.pdf"
save_path = documents_path + pdf_path.split('/')[-1].strip('.pdf') + '.txt'
extractor = Extract(pdf_path,device='cuda:1',method=method)

#docTXT = extractor.getTextFromImg(pdf_path)

if method == 'ocr':
    docTXT = extractor.getTextFromImg()
elif method == 'text':
    docTXT = extractor.getText()
    if len(docTXT)<50:
        extractor = Extract(pdf_path,device='cuda:2',method='ocr')
        docTXT = extractor.getTextFromImg()
with open(save_path, "w") as text_file:
    text_file.write(docTXT)

file_path = save_path

CH = RAGRetriever(file_path,device='cuda',
              embed_device='cuda',
              vecStoreDir = './',

              GeneratorModel='llama3.2:3b',
              temp=0)
query = 'what are the certified values' ##your query here
hint='ouput the values as json' # if needed, this is only passed to the generation model and not in retrieval (embedding search)
chunk_size = 1000 # how long is each chunk (carachter length), adjust to your case
chunk_overlap =  500 # overlap between chunks, adjust to your case
if vector_store_created == False:
    CH.createVecStore_multiVec(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    vector_store_created = True
answer = CH.mRetriever(Q=query,hint=hint)
