


document.addEventListener('DOMContentLoaded', () => {
    const homeLink = document.querySelector('a[href="#home"]');
    const aboutLink = document.querySelector('a[href="#about"]');
    const contactLink = document.querySelector('a[href="#contact"]');

    homeLink.addEventListener('click', () => toggleContent('home-content'));
    aboutLink.addEventListener('click', () => toggleContent('about-content'));
    contactLink.addEventListener('click', () => toggleContent('contact-content'));
});
// Get the modal
var loginModal = document.getElementById("loginModal");

// Get the button that opens the modal
var loginBtn = document.getElementById("loginLink");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks on the button, open the modal 
loginBtn.onclick = function() {
    loginModal.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
    loginModal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
    if (event.target == loginModal) {
        loginModal.style.display = "none";
    }
}


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


function sendRequest() {
    const userInput = document.getElementById("user-input").value;
    fetch('http://localhost:5000/generate_response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },  // Specify JSON content
        body: JSON.stringify({ input: userInput }) 
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("response-area").textContent = data.response;
    });
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
const models = [
    "AI-Driven Analysis For Water Suitability",
    "Depth Prediction For Water-Bearing Zones",
    "Predictive Well Discharge Calculations",
    "Recommendations For Drilling Techniques",
    "Groundwater Quality Forecasts"
  ];
  
  let currentIndex = 0;
  
  function showNextModel() {
    // Get the container for model descriptions
    const container = document.getElementById('ai-model-descriptions');
    
    // Remove the old text
    container.innerHTML = '';
    
    // Create the new text element
    const newText = document.createElement('div');
    newText.textContent = models[currentIndex];
    newText.classList.add('model-description');
    
    // Append the new text to the container
    container.appendChild(newText);
    
    // Fade in the text
    setTimeout(() => newText.classList.add('visible'), 100);
    
    // Prepare the next index, looping back to the start if necessary
    currentIndex = (currentIndex + 1) % models.length;
    
    // Set up the next change
    setTimeout(showNextModel, 4000); // Change text every 4 seconds
  }
  
  // Start the loop
  showNextModel();
  
