import json
import xml.etree.ElementTree as ET
from xml.dom import minidom


    
def writeFile(dictionary,out_path,file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    def modify_xml_entry(entry_value, entry_xpath, output_file=None):
        # Load the XML file


        element = root.find(entry_xpath)
        element.text = entry_value

        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"XML saved to '{output_file}'")

    def read_json_file(file_path='./questions_comprehensiveRAG.json'):
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                return data

    ID_paths = read_json_file('./questions_comprehensiveRAG.json')
    
    def getXpath(XML_ID, ID_paths):
        for id in ID_paths:
            if XML_ID == id['XML_ID']:
                 break
        return id['XML_path']
    
    XML_IDs = list(dictionary.keys())
    for xmlID in XML_IDs:
         Xpaths = getXpath(xmlID, ID_paths)
         for Xpath in Xpaths:
            if xmlID == 'respPersons':
                pathx = "/drmd:digitalReferenceMaterialDocument/drmd:administrativeData/drmd:respPersons/dcc:respPerson[1]/dcc:person/dcc:name/dcc:content"
                personsDict = json.loads(dictionary[xmlID])
                element = root.find(pathx)
                element.text = dictionary[xmlID]
