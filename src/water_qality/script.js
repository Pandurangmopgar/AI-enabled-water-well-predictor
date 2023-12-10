// script.js
document.addEventListener('DOMContentLoaded', function() {
    // Initialize your map here and get the map object
    // For example, if you're using Leaflet:
    var map = L.map('map').setView([51.505, -0.09], 13);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Placeholder for the coordinates
    var coords = {};

    // Map click event
    map.on('click', function(e) {
        coords = e.latlng;
        L.marker([coords.lat, coords.lng]).addTo(map);
        // You can also update the coords in a form input if you want to display them
    });

    // Prediction button event listener
    document.getElementById('predictButton').addEventListener('click', function() {
        if (!coords.lat || !coords.lng) {
            alert('Please select a location on the map.');
            return;
        }

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(coords)
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('predictionResult').textContent = data.prediction;
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });
});
