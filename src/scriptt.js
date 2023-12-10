// script.js

document.addEventListener('DOMContentLoaded', () => {
    const homeLink = document.querySelector('a[href="#home"]');
    const aboutLink = document.querySelector('a[href="#about"]');
    const contactLink = document.querySelector('a[href="#contact"]');

    homeLink.addEventListener('click', () => toggleContent('home-content'));
    aboutLink.addEventListener('click', () => toggleContent('about-content'));
    contactLink.addEventListener('click', () => toggleContent('contact-content'));
});

function toggleContent(sectionId) {
    // Hide all sections
    document.querySelectorAll('section').forEach(section => {
        section.style.display = 'none';
    });

    // Show the selected section
    const sectionToShow = document.getElementById(sectionId);
    if (sectionToShow) {
        sectionToShow.style.display = 'block';
    }
}



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
