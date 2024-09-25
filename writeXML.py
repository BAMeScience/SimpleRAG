import xml.etree.ElementTree as ET
from xml.dom import minidom
import uuid
import json




# Function to generate the XML structure
def create_xml(json_file,out_path):
    # Create the root element and its namespaces
    drmd_ns = "https://example.org/drmd"
    dcc_ns = "https://ptb.de/dcc"
    si_ns = "https://ptb.de/si"
    
    ET.register_namespace("drmd", drmd_ns)
    ET.register_namespace("dcc", dcc_ns)
    ET.register_namespace("si", si_ns)
    
    # Create root element
    root = ET.Element(f"{{{drmd_ns}}}digitalReferenceMaterialDocument", {
        "schemaVersion": "0.0.1"
    })

    # Create administrativeData -> coreData structure
    administrative_data = ET.SubElement(root, f"{{{drmd_ns}}}administrativeData")
    core_data = ET.SubElement(administrative_data, f"{{{drmd_ns}}}coreData")

    # Title of the document
    ET.SubElement(core_data, f"{{{drmd_ns}}}titleOfTheDocument").text = "Reference Material Certificate"

    # Get user input for uniqueIdentifier, issuer, value, etc.
    unique_identifier = str(uuid.uuid4())

    issuer = json_file['producer']
    value = json_file['MaterialCode']
    document_name = 'Certified Reference Material'
    period_of_validity = json_file['periodOfValidity']
    date_of_issue = json_file['dataOfIssue']

    ET.SubElement(core_data, f"{{{drmd_ns}}}uniqueIdentifier").text = unique_identifier
    identifications = ET.SubElement(core_data, f"{{{drmd_ns}}}identifications")
    identification = ET.SubElement(identifications, f"{{{drmd_ns}}}identification", {"refType": "basic_certificateIdentification"})
    ET.SubElement(identification, f"{{{drmd_ns}}}issuer").text = issuer
    ET.SubElement(identification, f"{{{drmd_ns}}}value").text = value
    name_elem = ET.SubElement(identification, f"{{{drmd_ns}}}name")
    ET.SubElement(name_elem, f"{{{dcc_ns}}}content", {"lang": "en"}).text = document_name

    ET.SubElement(core_data, f"{{{drmd_ns}}}periodOfValidity").text = period_of_validity
    ET.SubElement(core_data, f"{{{drmd_ns}}}dataOfIssue").text = date_of_issue

    # Add items
    items = ET.SubElement(administrative_data, f"{{{drmd_ns}}}items")
    item = ET.SubElement(items, f"{{{drmd_ns}}}item")

    item_name = 'XXXXXX'
    item_description = json_file['desctiption']

    name_elem = ET.SubElement(item, f"{{{drmd_ns}}}name")
    ET.SubElement(name_elem, f"{{{dcc_ns}}}content", {"lang": "en"}).text = item_name

    description_elem = ET.SubElement(item, f"{{{drmd_ns}}}description")
    ET.SubElement(description_elem, f"{{{dcc_ns}}}content", {"lang": "en"}).text = item_description

    # Get user input for valueXMLList and unitXMLList
    try:
        minimumSampleSize = json.loads(json_file['minimumSampleSize'])
        sample_size_value = str(minimumSampleSize['value'])
        sample_size_unit =  minimumSampleSize['Unit']
    except:
        sample_size_value = 'not found'
        sample_size_unit =  'not found'      

    # Minimum sample size (values taken from user input)
    minimum_sample_size = ET.SubElement(item, f"{{{drmd_ns}}}minimumSampleSize")
    item_quantity = ET.SubElement(minimum_sample_size, f"{{{dcc_ns}}}itemQuantity")
    real_list = ET.SubElement(item_quantity, f"{{{si_ns}}}realListXMLList")
    
    # User-provided values for sample size and unit
    ET.SubElement(real_list, f"{{{si_ns}}}valueXMLList").text = sample_size_value
    ET.SubElement(real_list, f"{{{si_ns}}}unitXMLList").text = sample_size_unit

    # More identifications (issuer and value)
    item_issuer = json_file['producer']
    item_value = 'XXXXX'

    identifications = ET.SubElement(item, f"{{{drmd_ns}}}identifications")
    identification = ET.SubElement(identifications, f"{{{drmd_ns}}}identification")
    ET.SubElement(identification, f"{{{drmd_ns}}}issuer").text = item_issuer
    ET.SubElement(identification, f"{{{drmd_ns}}}value").text = item_value

    # Add referenceMaterialProducer
    producer_name = json_file['producer']
    try:
        contact_details = json.loads(json_file['Contact'])

        contact_name = contact_details['Name']
        contact_email = contact_details['Email']
        contact_phone = str(contact_details['Phone'])
        contact_fax = str(contact_details['Fax'])
        contact_street = contact_details['street']
        contact_street_no = str(contact_details['streetNo'])
        contact_postcode = str(contact_details['postCode'])
        contact_city = contact_details['city']
        contact_country = contact_details['country']
    
    except:
        contact_name = 'not found'
        contact_email = 'not found'
        contact_phone = 'not found'
        contact_fax = 'not found'
        contact_street = 'not found'
        contact_street_no = 'not found'
        contact_postcode = 'not found'
        contact_city = 'not found'
        contact_country = 'not found'
        


    producer = ET.SubElement(administrative_data, f"{{{drmd_ns}}}referenceMaterialProducer")
    producer_name_elem = ET.SubElement(producer, f"{{{drmd_ns}}}name")
    ET.SubElement(producer_name_elem, f"{{{dcc_ns}}}content", {"lang": "en"}).text = producer_name

    contact = ET.SubElement(producer, f"{{{drmd_ns}}}contact")
    contact_name_elem = ET.SubElement(contact, f"{{{dcc_ns}}}name")
    ET.SubElement(contact_name_elem, f"{{{dcc_ns}}}content", {"lang": "en"}).text = contact_name
    ET.SubElement(contact, f"{{{dcc_ns}}}eMail").text = contact_email
    ET.SubElement(contact, f"{{{dcc_ns}}}phone").text = contact_phone
    ET.SubElement(contact, f"{{{dcc_ns}}}fax").text = contact_fax

    location = ET.SubElement(contact, f"{{{dcc_ns}}}location")
    ET.SubElement(location, f"{{{dcc_ns}}}street").text = contact_street
    ET.SubElement(location, f"{{{dcc_ns}}}streetNo").text = contact_street_no
    ET.SubElement(location, f"{{{dcc_ns}}}postCode").text = contact_postcode
    ET.SubElement(location, f"{{{dcc_ns}}}city").text = contact_city
    ET.SubElement(location, f"{{{dcc_ns}}}countryCode").text = contact_country




    # Add respPersons section
    resp_persons = ET.SubElement(administrative_data, f"{{{drmd_ns}}}respPersons")
    try:
        persons = json.loads(json_file['respPersons'].replace("'",'"'))
        persons_names = list(persons.keys())
        persons_roles = list(persons.values())
        for i in range(len(persons_names)):  # Assuming two responsible persons
            resp_person = ET.SubElement(resp_persons, f"{{{dcc_ns}}}respPerson")
            person = ET.SubElement(resp_person, f"{{{dcc_ns}}}person")
            person_name = persons_names[i]
            person_role = persons_roles[i]
            name_elem = ET.SubElement(person, f"{{{dcc_ns}}}name")
            ET.SubElement(name_elem, f"{{{dcc_ns}}}content", {"lang": "en"}).text = person_name
            ET.SubElement(person, f"{{{dcc_ns}}}role").text = person_role

    except:
        for i in range(1):  # Assuming two responsible persons
            resp_person = ET.SubElement(resp_persons, f"{{{dcc_ns}}}respPerson")
            person = ET.SubElement(resp_person, f"{{{dcc_ns}}}person")
            person_name = 'not found'
            person_role = 'not found'
            name_elem = ET.SubElement(person, f"{{{dcc_ns}}}name")
            ET.SubElement(name_elem, f"{{{dcc_ns}}}content", {"lang": "en"}).text = person_name
            ET.SubElement(person, f"{{{dcc_ns}}}role").text = person_role
    # Add statements section
    statements = ET.SubElement(administrative_data, f"{{{drmd_ns}}}statements")

    intended_use = ET.SubElement(statements, f"{{{drmd_ns}}}intendedUse")
    ET.SubElement(ET.SubElement(intended_use, f"{{{dcc_ns}}}name"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = "Intended Use"
    intended_use_desc = json_file['IntendedUse']
    ET.SubElement(ET.SubElement(intended_use, f"{{{dcc_ns}}}description"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = intended_use_desc

    storage_info = ET.SubElement(statements, f"{{{drmd_ns}}}storageInformation")
    ET.SubElement(ET.SubElement(storage_info, f"{{{dcc_ns}}}name"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = "Storage Information"
    storage_info_desc = json_file['StorageInformation']
    ET.SubElement(ET.SubElement(storage_info, f"{{{dcc_ns}}}description"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = storage_info_desc

    handling_instr = ET.SubElement(statements, f"{{{drmd_ns}}}instructionsForHandlingAndUse")
    ET.SubElement(ET.SubElement(handling_instr, f"{{{dcc_ns}}}name"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = "Handling Instructions"
    handling_instr_desc = json_file['instructionsForHandlingAndUse']
    ET.SubElement(ET.SubElement(handling_instr, f"{{{dcc_ns}}}description"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = handling_instr_desc

    metrological_traceability = ET.SubElement(statements, f"{{{drmd_ns}}}metrologicalTraceability")
    ET.SubElement(ET.SubElement(metrological_traceability, f"{{{dcc_ns}}}name"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = "Metrological Traceability"
    metrological_traceability_desc = json_file['metrologicalTraceability']
    ET.SubElement(ET.SubElement(metrological_traceability, f"{{{dcc_ns}}}description"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = metrological_traceability_desc

    subcontractors = ET.SubElement(statements, f"{{{drmd_ns}}}subcontractors")
    ET.SubElement(ET.SubElement(subcontractors, f"{{{dcc_ns}}}name"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = "Subcontractors"
    subcontractors_desc = json_file['subcontractors']
    ET.SubElement(ET.SubElement(subcontractors, f"{{{dcc_ns}}}description"), f"{{{dcc_ns}}}content", {"lang": "en"}).text = subcontractors_desc

    # Add measurementResults section
    measurement_results = ET.SubElement(root, f"{{{drmd_ns}}}measurementResults")
    measurement_result = ET.SubElement(measurement_results, 'drmd:measurementResult', {'isCertified': 'true'})

    # Create dcc:name element with content
    name = ET.SubElement(measurement_result, 'dcc:name')
    content_name = ET.SubElement(name, 'dcc:content', {'lang': 'en'})
    content_name.text = 'Certified mass fractions and their associated uncertainties.'

    # Create dcc:results element
    results = ET.SubElement(measurement_result, 'dcc:results')

    # Create dcc:result element
    result = ET.SubElement(results, 'dcc:result')

    # Create dcc:name element inside result
    result_name = ET.SubElement(result, 'dcc:name')
    result_content_name = ET.SubElement(result_name, 'dcc:content', {'lang': 'en'})
    result_content_name.text = 'Certified Values'

    # Create dcc:description element inside result
    description = ET.SubElement(result, 'dcc:description')
    description_content = ET.SubElement(description, 'dcc:content', {'lang': 'en'})
    description_content.text = json_file['certifiedValues_Footnote']

    # Create dcc:data element inside result
    data = ET.SubElement(result, 'dcc:data')

    # Create dcc:list element inside data
    data_list = ET.SubElement(data, 'dcc:list')
    try:
        certValues = json.loads(json_file['certifiedValues'])
        len_certValues = len(certValues)
    except Exception as e:
        len_certValues = 0
        print("""Error accured while writing table:""", e)

    try:
        U_parameters = json.loads(json_file['UncertaintyParameters'])
        coverageFactor = str(U_parameters['coverage factor'])
        confidence = str(U_parameters['confidence'])

    except Exception as e:
        coverageFactor = 'not found'
        confidence = 'not found'
    
    for i in range(len_certValues):
        quantity = ET.SubElement(data_list, 'dcc:quantity', {'refType': 'basic_measuredValue'})

        # Create dcc:name element inside quantity
        quantity_name = ET.SubElement(quantity, 'dcc:name')
        quantity_content_name = ET.SubElement(quantity_name, 'dcc:content', {'lang': 'en'})
        quantity_content_name.text = certValues[i]['measurand']

        # Create si:real element inside quantity
        real = ET.SubElement(quantity, 'si:real')

        # Create si:label element inside real
        label = ET.SubElement(real, 'si:label')
        label.text = certValues[i]['measurand']

        # Create si:value element inside real
        value = ET.SubElement(real, 'si:value')
        value.text = str(certValues[i]['mass fraction'])

        # Create si:unit element inside real
        unit = ET.SubElement(real, 'si:unit')
        unit.text = str(certValues[i]['mass fraction unit'])

        # Create si:expandedUnc element inside real
        expanded_unc = ET.SubElement(real, 'si:expandedUnc')

        # Create si:uncertainty inside si:expandedUnc
        uncertainty = ET.SubElement(expanded_unc, 'si:uncertainty')
        uncertainty.text = str(certValues[i]['uncertainty'])

        # Create si:coverageFactor inside si:expandedUnc
        coverage_factor = ET.SubElement(expanded_unc, 'si:coverageFactor')
        coverage_factor.text = coverageFactor

        # Create si:coverageProbability inside si:expandedUnc
        coverage_probability = ET.SubElement(expanded_unc, 'si:coverageProbability')
        coverage_probability.text = confidence

        # Create si:distribution inside si:expandedUnc
        distribution = ET.SubElement(expanded_unc, 'si:distribution')
        distribution.text = 'normal'
    # Convert the ElementTree to a string and pretty print
    xml_str = ET.tostring(root, encoding='utf-8')
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="   ")

    # Write to an XML file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml_str)

    print("XML file created successfully!")



