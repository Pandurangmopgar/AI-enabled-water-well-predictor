document.getElementById('wellForm').addEventListener('submit', function(event) {
    event.preventDefault();
    let modelSelection = document.getElementById('modelSelection').value;
    // Process the selected model and display results
    // This is where you'll integrate with your backend system
    document.getElementById('result').innerHTML = 'Results for model: ' + modelSelection;
});

document.getElementById('feedbackForm').addEventListener('submit', function(event) {
    event.preventDefault();
    let feedback = document.getElementById('userFeedback').value;
    // Process the feedback
    // You may want to send this data to your server
    alert('Thank you for your feedback!');
});
