const fileInputs = {
    1: ['TCRα', 'TCRβ', 'Peptide', 'MHC'],
    2: ['TCRα', 'TCRβ', 'Peptide'],
    3: ['TCRα', 'Peptide', 'MHC'],
    4: ['TCRβ', 'Peptide', 'MHC'],
    5: ['TCRα', 'Peptide'],
    6: ['TCRβ', 'Peptide']
  };
  
  function renderFileInputs(task) {
    const container = document.getElementById("fileInputs");
    container.innerHTML = "";
  
    fileInputs[task].forEach((name, idx) => {
      const div = document.createElement("div");
      div.classList.add("file-group");
      div.innerHTML = `
        <label>Choose the .npy file containing ${name} Embeddings</label>
        <input type="file" name="file${idx+1}" accept=".npy" required />
      `;
      container.appendChild(div);
    });
  }
  
  document.querySelectorAll('input[name="task"]').forEach(input => {
    input.addEventListener('change', (e) => renderFileInputs(e.target.value));
  });
  
  // Initial render
  renderFileInputs("1");
  