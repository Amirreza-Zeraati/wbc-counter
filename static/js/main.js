let selectedFile = null;

const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const processBtn = document.getElementById('processBtn');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const error = document.getElementById('error');
const resetBtn = document.getElementById('resetBtn');

// Click to upload
uploadBox.addEventListener('click', () => fileInput.click());

// File selection
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    selectedFile = file;
    uploadBox.querySelector('.upload-text').textContent = `Selected: ${file.name}`;
    processBtn.disabled = false;
    hideError();
}

// Process button
processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    showLoader();
    hideError();
    results.style.display = 'none';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            showResults(data);
        }
    } catch (err) {
        showError('An error occurred during processing');
    } finally {
        hideLoader();
    }
});

// Reset button
resetBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadBox.querySelector('.upload-text').textContent = 'Click to upload or drag and drop';
    processBtn.disabled = true;
    results.style.display = 'none';
    hideError();
});

function showLoader() {
    loader.style.display = 'block';
    processBtn.disabled = true;
}

function hideLoader() {
    loader.style.display = 'none';
    processBtn.disabled = false;
}

function showResults(data) {
    document.getElementById('originalImg').src = data.original;
    document.getElementById('processedImg').src = data.processed;
    document.getElementById('cellCount').textContent = data.count;
    results.style.display = 'block';
}

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}
