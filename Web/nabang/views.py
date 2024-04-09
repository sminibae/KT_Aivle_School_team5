from django.shortcuts import render
from django.http import JsonResponse

def index(request):
    return render(request, 'index.html')

def airecommend(request):
    return render(request, 'airecommend.html')

def file_upload(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        
        # 파일 처리 로직 작성
        # 예: 파일을 서버에 저장하고, 처리 결과를 반환

        return JsonResponse({'status': 'success', 'filename': uploaded_file.name})
    return JsonResponse({'status': 'error'}, status=400)