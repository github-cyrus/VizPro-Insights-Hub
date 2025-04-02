const firebaseConfig = {
  //   copy your firebase config informations
  apiKey: "AIzaSyBWRHIx0Vvo3UrcF-wcmSe99VALu9e0iiM",
  authDomain: "vizpro-9dce2.firebaseapp.com",
  databaseURL: "https://vizpro-9dce2-default-rtdb.firebaseio.com",
  projectId: "vizpro-9dce2",
  storageBucket: "vizpro-9dce2.appspot.com",
  messagingSenderId: "440136532551",
  appId: "1:440136532551:web:f5980b3c17ef65b10b6762"
};

// initialize firebase
firebase.initializeApp(firebaseConfig);

// reference your database
var contactFormDB = firebase.database().ref("contact");

document.getElementById("contact").addEventListener("submit", submitForm);

function submitForm(e) {
  e.preventDefault();

  var name = getElementVal("name");
  var email = getElementVal("email");
  var subject = getElementVal("subject");
  var message = getElementVal("message");

  saveMessages(name, email, subject, message);

  //   enable alert
  document.querySelector(".alert").style.display = "block";

  //   remove the alert
  setTimeout(() => {
    document.querySelector(".alert").style.display = "none";
  }, 3000);

  //   reset the form
  document.getElementById("contact").reset();
}

const saveMessages = (name, email, subject,message) => {
  var newContactForm = contactFormDB.push();

  newContactForm.set({
    name: name,
    emailid: email,
    subject: subject,
    message: message,
  });
};

const getElementVal = (id) => {
  return document.getElementById(id).value;
};
