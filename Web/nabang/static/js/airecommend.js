function previewFile(file) {
    const preview = document.getElementById('preview-image');
    const uploadInstructions = document.getElementById('upload-instructions');
    const changeInstructions = document.getElementById('change-instructions');
    const reader = new FileReader();

    reader.onloadend = function () {
        preview.src = reader.result;
        preview.style.display = 'block';
        uploadInstructions.style.display = 'none';
        changeInstructions.style.display = 'block'; // 이미지 업로드 시 설명 표시
    }

    if (file) {
        reader.readAsDataURL(file);
    } else {
        preview.src = "";
        preview.style.display = 'none';
        uploadInstructions.style.display = 'block';
        changeInstructions.style.display = 'none'; // 이미지가 없을 때 설명 숨김
    }
}

document.getElementById('file-upload').addEventListener('change', function(event) {
    if (this.files && this.files[0]) {
        previewFile(this.files[0]);
    }
});

document.getElementById('drag-drop-area').addEventListener('dragover', function(event) {
    event.preventDefault();
    event.stopPropagation();
    this.classList.add('drag-over'); // 드래그 중 스타일 적용
}, false);

document.getElementById('drag-drop-area').addEventListener('dragleave', function(event) {
    event.preventDefault();
    event.stopPropagation();
    this.classList.remove('drag-over'); // 드래그 끝 스타일 제거
}, false);

document.getElementById('drag-drop-area').addEventListener('drop', function(event) {
    event.preventDefault();
    event.stopPropagation();
    this.classList.remove('drag-over'); // 드롭 후 스타일 제거

    const file = event.dataTransfer.files[0]; // 드롭된 파일
    document.getElementById('file-upload').files = event.dataTransfer.files; // input 필드에 파일 설정

    previewFile(file); // 미리보기 함수에 파일 전달
}, false);

document.getElementById('file-upload').addEventListener('change', function(event) {
    if (this.files && this.files[0]) {
        previewFile(this.files[0]);
    }
    updateAnalyzeButton(); // 파일이 변경될 때마다 버튼 상태 업데이트
});
function updateAnalyzeButton() {
    const fileInput = document.getElementById('file-upload');
    const analyzeButton = document.getElementById('analyze-button');
    analyzeButton.disabled = !fileInput.files.length;
}

// 파일 입력 변경 이벤트
document.getElementById('file-upload').addEventListener('change', function(event) {
    if (this.files && this.files[0]) {
        previewFile(this.files[0]);
    }
    updateAnalyzeButton(); // 버튼 상태 업데이트
});

// 드래그 앤 드롭 이벤트 핸들러
document.getElementById('drag-drop-area').addEventListener('drop', function(event) {
    event.preventDefault();
    event.stopPropagation();
    this.classList.remove('drag-over');

    const file = event.dataTransfer.files[0];
    document.getElementById('file-upload').files = event.dataTransfer.files;

    previewFile(file); // 미리보기
    updateAnalyzeButton(); // 드롭 후 버튼 상태 업데이트
}, false);
