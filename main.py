from writeXML import create_xml
from Ragger import RAGRetriever
import json


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return data

vector_store_created = False
#file_path = "./CRM_text/Certificate_BAM-U026.txt"
#file_path = "./CRM_text/Certificate_BAM_A001.txt"
#file_path = './CRM_text/578-2 Zert._englische Fassung_v3.txt'
file_path = './CRM_text/Zertifikat BAM-U117_V2.txt'
#file_path = './CRM_text/Zertifikat BAM-S006_V2_Unterschrift.txt'


file_name = file_path.split('/')[-1].strip('.txt')

CH = RAGRetriever(file_path,device='cuda',
              embed_device='cuda',
              vecStoreDir = './',
              GeneratorModel='llama3:70b-instruct',
              temp=0)
#CH.MultiRetriever.invoke('what is the description of the material')
questions = read_json_file('./questions_comprehensiveRAG.json')

results = []
resultsDict = {}
for QandH in questions:

    query = QandH['Question']
    hint = QandH['Hint']
    
    if vector_store_created == False:
        CH.createVecStore_multiVec(chunk_size=1000,chunk_overlap=500)
        vector_store_created = True
    Answer = CH.mRetriever(Q=query,hint=hint)
    #CH.MultiRetriever.invoke(query) # see retrieved documents/chunks
    results.append(Answer)
    resultsDict[QandH['XML_ID']] = Answer

print(resultsDict)
def formatting(res):
    keys = ['respPersons', 'Contact','minimumSampleSize','UncertaintyParameters']
    for k in keys:

        curl_idx = (res[k].find('{'),
                res[k].find('}'))

        res[k] = res[k][curl_idx[0]:curl_idx[1]+1]
    
    tables_keys = ['certifiedValues']
    for k in tables_keys:

        curl_idx = (res[k].find('[\n'),
                res[k].find('\n]'))

        res[k] = res[k][curl_idx[0]:curl_idx[1]+len('\n]')]
    return res


resultsDict_formatted = formatting(resultsDict)
for formattedResult in resultsDict_formatted:
    print(resultsDict_formatted[formattedResult])


save_path_json = "./retrieved/" + file_name + ".json"
save_path_xml = "./ExportedXML/" + file_name + ".xml"

def save_dict_as_json(dictionary, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dictionary, json_file, indent=4, ensure_ascii=False)
save_dict_as_json(resultsDict_formatted,save_path_json)




create_xml(resultsDict_formatted,save_path_xml)

