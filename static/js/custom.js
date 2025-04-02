// ...existing code...

// Fix navigation scroll
function visible(el) {
    if (!el || !el.offsetTop) return false;
    const rect = el.getBoundingClientRect();
    const viewHeight = Math.max(document.documentElement.clientHeight, window.innerHeight);
    return !(rect.bottom < 0 || rect.top - viewHeight >= 0);
}

// Fix gallery link
$(document).on('click', 'a[href="Dashboard Gallery/gallery.html"]', function(e) {
    e.preventDefault();
    window.location.href = '/gallery.html';
});

// ...existing code...