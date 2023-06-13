const imageInput = document.getElementById('imageInput');
const resultDiv = document.getElementById('result');

imageInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = (e) => {
        const img = new Image();
        img.src = e.target.result;
        img.onload = () => {
            classifyImage(img);
        };
    };

    reader.readAsDataURL(file);
});

function classifyImage(image) {
    console.log('Classifying image...');
    const formData = new FormData();
    formData.append('image', image);

    fetch('/classify', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const result = data.class_name;
        const class_names = data.class_names;

        // Display the classification result
        resultDiv.innerText = `This flower is ${result}`;

        // Output the class names to the console for verification
        console.log(class_names);
    })
    .catch(error => console.error(error));
}
