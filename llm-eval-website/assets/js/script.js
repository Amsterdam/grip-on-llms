
document.addEventListener('DOMContentLoaded', function() {
  const menuBtn = document.getElementById('menu-btn');
  const navMenu = document.getElementById('nav-menu');
  if (menuBtn && navMenu) {
    menuBtn.addEventListener('click', function() {
      navMenu.classList.toggle('active');
    });
  }

  // Open disclaimer on click
  document.querySelectorAll('.missing-score-icon').forEach(function(icon) {
    icon.addEventListener('click', function(e) {
      var card = icon.querySelector('.score-disclaimer-card');
      if(card) card.style.display = 'flex';
      e.stopPropagation();
    });
  });

  // Open explanation on click
  document.querySelectorAll('.score-info-icon').forEach(function(icon) {
    icon.addEventListener('click', function(e) {
      var card = icon.querySelector('.score-explanation-card');
      if(card) card.style.display = 'flex';
      e.stopPropagation();
    });
  });

  // Close buttons (both disclaimer and explanation)
  document.querySelectorAll('.score-disclaimer-close, .score-explanation-close').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      // Find the closest card (disclaimer or explanation)
      var card = btn.closest('.score-disclaimer-card, .score-explanation-card');
      if(card) card.style.display = 'none';
      e.stopPropagation();
    });
  });

  // Clicking outside any popup closes all open popups
  document.addEventListener('click', function(e) {
    document.querySelectorAll('.score-disclaimer-card, .score-explanation-card').forEach(function(card) {
      if(card.style.display === 'flex' && !card.contains(e.target)) {
        card.style.display = 'none';
      }
    });
  });
});