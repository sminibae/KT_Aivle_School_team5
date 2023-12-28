function showLoginPopup() {
    var popup = document.getElementById('login-popup');
    popup.style.display = 'flex'; // 팝업을 flex로 설정하여 보여줌
}

// 로그인 팝업 숨기기 함수
function hideLoginPopup() {
    document.getElementById('login-popup').style.display = 'none';
}

// 로그인 링크 이벤트 리스너
document.querySelector('.nav-login-link').addEventListener('click', function(event) {
    event.preventDefault();
    showLoginPopup();
});

// 팝업 닫기 버튼 이벤트 리스너
document.querySelector('.close-button').addEventListener('click', function() {
    hideLoginPopup(); // 로그인 팝업 숨김
});
