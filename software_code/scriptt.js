document.addEventListener('DOMContentLoaded', (event) => {
    // Select all links with the class 'ai-model'
    const aiModelLinks = document.querySelectorAll('.ai-model');

    // Add click event listener to each link
    aiModelLinks.forEach((link) => {
        link.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent the default link behavior
            const destination = link.getAttribute('href'); // Get the href value from the clicked link
            window.location.href = destination; // Redirect to the linked page
        });
    });
});
