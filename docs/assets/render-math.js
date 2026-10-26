// Deferred after KaTeX: render both language panels, including the hidden one.
// The readable HTML formula remains available if the CDN cannot be reached.
if (window.katex) {
    document.querySelectorAll('[data-math]').forEach(element => {
        try {
            element.innerHTML = window.katex.renderToString(element.dataset.math, {
                displayMode: element.dataset.mathDisplay !== 'inline',
                output: 'htmlAndMathml',
                throwOnError: true,
                trust: false,
            });
        } catch (error) {
            console.warn('Unable to typeset a formula; keeping its readable fallback.', error);
        }
    });
}
