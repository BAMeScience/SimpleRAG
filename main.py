from writeXML import create_xml
from Ragger import RAGRetriever
import json


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return data
    


vector_store_created = False
file_path = './CRM_text/Certificate_BAM_A001.txt'
file_name = file_path.split('/')[-1].strip('.txt')

CH = RAGRetriever(file_path,device='cuda',
              embed_device='cuda',
              vecStoreDir = './',
              GeneratorModel='llama3:instruct',
              temp=0)

questions = read_json_file('./questions_comprehensiveRAG.json')

results = []
resultsDict = {}
for QandH in questions:

    query = QandH['Question']
    hint = QandH['Hint']
    
    if vector_store_created == False:
        CH.createVecStore_multiVec(chunk_size=1500,chunk_overlap=500)
        vector_store_created = True
    Answer = CH.mRetriever(Q=query,hint=hint)
    #CH.MultiRetriever.invoke(query) # see retrieved documents/chunks
    results.append(Answer)
    resultsDict[QandH['XML_ID']] = Answer

print(resultsDict)

curl_idx = (resultsDict['respPersons'].find('{'),
            resultsDict['respPersons'].find('}'))

resultsDict['respPersons'] = resultsDict['respPersons'][curl_idx[0]:curl_idx[1]+1]



save_path_json = "./retrieved/" + file_name + ".json"
save_path_xml = "./ExportedXML/" + file_name + ".xml"

def save_dict_as_json(dictionary, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dictionary, json_file, indent=4, ensure_ascii=False)
        print(f"Dictionary saved to {file_path}")
save_dict_as_json(resultsDict,save_path_json)




create_xml(resultsDict,save_path_xml)

