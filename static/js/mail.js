// Initialize Firebase
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Get contact form element
const contactForm = document.getElementById('contact');

// Handle form submission
contactForm.addEventListener('submit', (e) => {
  e.preventDefault();

  const name = document.getElementById('name').value;
  const email = document.getElementById('email').value;
  const subject = document.getElementById('subject').value; 
  const message = document.getElementById('message').value;

  // Save to Firebase
  firebase.database().ref('contacts').push({
    name: name,
    email: email,
    subject: subject,
    message: message,
    timestamp: Date.now()
  })
  .then(() => {
    alert('Message sent successfully!');
    contactForm.reset();
  })
  .catch(error => {
    alert('Error sending message: ' + error.message);
  });
});
