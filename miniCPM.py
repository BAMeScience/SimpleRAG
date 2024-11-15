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
        for page in images:
            self.f = page

            question= """
                    Write down everything in the image as text.
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





