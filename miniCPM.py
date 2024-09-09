import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_path
import pathlib
from PyPDF2 import PdfReader
class Extract():
    def __init__(self,pdf_path, device = 'cuda',method='text'):
        if method=='ocr':
            self.model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)#, torch_dtype=torch.bfloat16)
            self.model = self.model.to(device=device)#, dtype=torch.bfloat16)


            self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
            self.model.eval()
        else:
            pass
        self.path = pdf_path
    def getText(self):
        reader = PdfReader(self.path)
        number_of_pages = len(reader.pages)
        text = ''
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = text + page.extract_text() + 'new page \n '
        return text
    def getTextFromImg(self):
        if pathlib.Path(self.path).suffix == '.pdf':
            images = convert_from_path(self.path, dpi=800)
        else:
            images = [Image.open(self.path).convert('RGB')]
        documentTXT = ''
        context = """
                    Instructions:
                    1. Write down everything in the image as text.
                    2. Tables should be written as dictionaries in the following format:
                    {
                        "Column 1": ["Row1Value1", "Row2Value1", ...],
                        "Column 2": ["Row1Value2", "Row2Value2", ...],
                        ...
                    }
                    3. Units must also be included in the table, if found.
                    4. Add the string '---Sec---' before each section.
                    5. Don't ignore any text.
                    Please adhere to these instructions strictly.
                """
        context = """You are an expert at extracting every single detail in tables."""
        for page in images:
            self.f = page
            '''
            question = "write down everything in the image as text,\
                  tables should be written in markdown using '|' as a seperator for exmaple,\
                  you MUST add the special phrase '[Table Start]' and '[Table End]' before and after table,\
                  '[Table Start]' and '[Table End]' must include everything related to the table,\
                  such as Title, caption or references if any is found.\
                      Dont ignore any text."
            '''
            question = "write down everything in the image as text,\
                tables should be written as dict,\
                units must also be included in the table for each cell.\
                add the string '---Sec---' before each section.\
                Dont ignore any text."
            question =  """
                Write down everything in the image as text.
                Don't ignore any text. Seperate sections with '---Sec---'.
                """
            question= """
                    Write down everything in the image as text.
                """
            question= """
                    Write down everything in the image as text.
                    Write tables in markdown with column seperators '|'.
                """
            #question= """Write down the tables in the image with units for each cell.
            #Write down everything in the image with seperators."""
            msgs = [{'role': 'user', 'content': question}]

            res= self.model.chat(
                image=page,
                msgs=msgs,
                context=None,
                tokenizer=self.tokenizer,
                sampling=False,
                temperature=0.7
            )
            
            documentTXT= documentTXT + '\n' + res + 'new page \n'
        return documentTXT

methods = ['ocr','text']
method = methods[1]

#pdf_path = "./Certificate_BAM_A001.pdf"
#pdf_path = "/home/balbakri/ragger/Zertifikat BAM-U117_V2.pdf"
#pdf_path = '/home/balbakri/ragger/Screenshot from 2024-07-25 15-24-19.png'
pdf_path = '/home/balbakri/ragger/first_model/CRM/Zertifikat BAM-U117_V2.pdf'
save_path = "./CRM_text/" + pdf_path.split('/')[-1].strip('.pdf') + '.txt'
extractor = Extract(pdf_path,device='cuda:1',method=method)

#docTXT = extractor.getTextFromImg(pdf_path)

if method == 'ocr':
    docTXT = extractor.getTextFromImg()
elif method == 'text':
    docTXT = extractor.getText()
    if len(docTXT)<100:
        extractor = Extract(pdf_path,device='cuda:2',method='ocr')
        docTXT = extractor.getTextFromImg()
with open(save_path, "w") as text_file:
    text_file.write(docTXT)

