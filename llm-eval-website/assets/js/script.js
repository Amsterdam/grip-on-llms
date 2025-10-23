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

  // TABLE SORTING - ALL COLUMNS
  const table = document.querySelector('table');
  if (table) {
    const headers = table.querySelectorAll('th');
    const tbody = table.querySelector('tbody');

    headers.forEach(function(header, index) {
      // Make all headers sortable
      header.style.cursor = 'pointer';
      header.classList.add('sortable');

      // Add click handler, but prevent sorting when clicking info icon
      header.addEventListener('click', function(e) {
        // Don't sort if clicking on the info icon or its children
        if (e.target.closest('.score-info-icon') || e.target.closest('.missing-score-icon')) {
          return;
        }
        sortTable(index, header);
      });
    });

    function sortTable(columnIndex, header) {
      const rows = Array.from(tbody.querySelectorAll('tr'));

      const isFirstSort = !header.classList.contains('sort-asc') && !header.classList.contains('sort-desc');
      const defaultSort = header.getAttribute('data-default-sort') || 'asc';
      console.log("defaultSort", defaultSort);

      let needAscending;

      if (isFirstSort) {
        // First click: check data-default-sort attribute
        needAscending = (defaultSort === 'asc');
      } else {
        // Toggle on subsequent clicks
        needAscending = !header.classList.contains('sort-asc');
      }
      console.log("Need ascending", needAscending);

      // Remove sort classes from all headers
      headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));

      // Add appropriate class to current header
      header.classList.add(needAscending ? 'sort-asc' : 'sort-desc');

      rows.sort(function(a, b) {
        const cellA = a.querySelectorAll('td')[columnIndex];
        const cellB = b.querySelectorAll('td')[columnIndex];

        let valueA = getCellValue(cellA);
        let valueB = getCellValue(cellB);

        // Treat empty/missing values as highest (push to bottom)
        const isEmptyA = !valueA || valueA === '' || valueA === '?' || valueA === '-1';
        const isEmptyB = !valueB || valueB === '' || valueB === '?' || valueB === '-1';

        if (isEmptyA && isEmptyB) return 0;  // Both empty, equal
        if (isEmptyA) return 1;   // A is empty, push down (after B)
        if (isEmptyB) return -1;  // B is empty, push down (after A)

        // Handle numeric values (including € costs like "€0.08")
        const numA = parseFloat(valueA.replace(/[€,]/g, ''));
        const numB = parseFloat(valueB.replace(/[€,]/g, ''));

        if (!isNaN(numA) && !isNaN(numB)) {
          return needAscending ? numA - numB : numB - numA;
        }

        // Handle text values
        return needAscending ?
          valueA.localeCompare(valueB) :
          valueB.localeCompare(valueA);
      });

      // Reorder rows in DOM
      rows.forEach(row => tbody.appendChild(row));
    }

    function getCellValue(cell) {
      if (!cell) return '';

      // Check for data-sort-value attribute first
      if (cell.hasAttribute('data-sort-value')) {
        return cell.getAttribute('data-sort-value');
      }

      // Try to get text from links first
      const link = cell.querySelector('a');
      if (link) return link.textContent.trim();

      // Check for costs bar (has data in text)
      const costsText = cell.textContent.trim();
      if (costsText) return costsText;

      // Otherwise get all text content
      return cell.textContent.trim();
    }
  }
});
