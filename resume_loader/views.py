from django.shortcuts import render
from django.http import JsonResponse
import PyPDF2
import io
from django.views.decorators.csrf import csrf_exempt
from docx import Document
import json
from resume_loader.llm import ask_llm

# Create your views here.
@csrf_exempt
def UploadResume(request):
    if request.method == 'POST':
        # 1. Check if the file exists in the request
        if 'pdf_file' not in request.FILES:
            print("no pdf detected")
            return JsonResponse({'error': 'No file uploaded'}, status=400)
            

        pdf_file = request.FILES['pdf_file']

        # 2. Validate file type
        if not pdf_file.name.endswith('.pdf') and not pdf_file.name.endswith('.docx'):
            
            return JsonResponse({'error': 'File is not a PDF or DOCX'}, status=400)

        try:
            # 3. Read the PDF content from memory
            # We wrap the uploaded bytes in io.BytesIO so PyPDF2 can read it like a file
            if pdf_file.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
                
                extracted_text = ""
                
                # 4. Loop through pages and extract text
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_text += page.extract_text() + "\n"

                # 5. Return the extracted text as JSON
                return JsonResponse({
                    'message': 'File processed successfully',
                    'filename': pdf_file.name,
                    'text': extracted_text
                }, status=200)
            elif pdf_file.name.endswith('.docx'):
                file_stream = io.BytesIO(pdf_file.read())
                doc = Document(file_stream)
                text_content = [para.text for para in doc.paragraphs if para.text.strip()]
                return JsonResponse({
                    'message': 'File processed successfully',
                    'filename': pdf_file.name,
                    'text': ''.join(text_content)
                }, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    print("Invalid request method")

    return JsonResponse({'error': 'Invalid request method'}, status=405)

#Resume Analyze function starts from here
@csrf_exempt
def AnalyzeResume(request):
    if request.method == 'POST':
        body=json.loads(request.body)
        resume_text = body.get('resume_text')
        job_description = body.get("job_description")
        messages = [{'role':'user','content':f"""
                resume text:{resume_text},
                job description:{job_description}
                """}]
        resume_analysis=ask_llm(messages)
        result = json.loads(resume_analysis)
        return JsonResponse({'message':'recieved successfully',
                             'response':result
                             })
