/**
 * IBM Financial Analysis Assistant
 * Handles file uploads and analysis for Watson Assistant integration
 */

// Counter for generating unique element IDs
let uploadCounter = 0;
let currentFilePath = null;
let analysisInProgress = false;

/**
 * Creates the file upload button inside the Watson Assistant chat interface
 */
function fileUploadCustomResponseHandler(event, instance) {
  uploadCounter++;
  const uniqueId = `upload_${Date.now()}_${uploadCounter}`;
  const { element } = event.data;
  
  element.innerHTML = `
    <div>
      <input type="file" id="uploadInput_${uniqueId}" style="display: none;" accept=".pdf">
      <button id="uploadButton_${uniqueId}" class="WAC__button--primary WAC__button--primaryMd">
        Upload PDF File
      </button>
      <div id="uploadStatus_${uniqueId}" style="margin-top: 5px; font-size: 12px;"></div>
    </div>
  `;
  
  const uploadInput = element.querySelector(`#uploadInput_${uniqueId}`);
  const button = element.querySelector(`#uploadButton_${uniqueId}`);
  const statusEl = element.querySelector(`#uploadStatus_${uniqueId}`);

  button.addEventListener("click", () => {
    uploadInput.click();
  });

  uploadInput.addEventListener("change", (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) {
      return;
    }
    
    // Check if file is PDF
    if (!selectedFile.type.match('application/pdf')) {
      statusEl.innerHTML = '<span style="color: #e62325;">Only PDF files are supported</span>';
      return;
    }
    
    // Check file size (limit to 50MB)
    if (selectedFile.size > 50 * 1024 * 1024) {
      statusEl.innerHTML = '<span style="color: #e62325;">File size exceeds 50MB limit</span>';
      return;
    }
    
    statusEl.innerHTML = '<span style="color: #0f62fe;">Uploading...</span>';
    uploadFileFromAsst(selectedFile, statusEl);
  });
}

/**
 * Handles the file upload process and sends to backend
 */
function uploadFileFromAsst(selectedFile, statusEl) {
  if (!selectedFile) {
    console.error("No file selected.");
    return;
  }
  
  // Prevent multiple simultaneous analysis runs
  if (analysisInProgress) {
    if (statusEl) statusEl.innerHTML = '<span style="color: #e62325;">Analysis already in progress, please wait</span>';
    return;
  }
  
  analysisInProgress = true;
  
  const formData = new FormData();
  formData.append("uploaded_file", selectedFile);
  
  // Always use the current origin for API calls when using Flask's built-in server
  const SERVER = window.location.origin;

  console.log("Uploading to server:", SERVER);
  
  // Show loading states in all analysis sections
  showLoadingStateForAll();
  
  fetch(SERVER + "/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      console.log("HTTP status:", response.status);
      if (response.ok) {
        if (statusEl) statusEl.innerHTML = '<span style="color: green;">Upload successful! Starting analysis...</span>';
        return response.json();
      } else {
        analysisInProgress = false;
        hideLoadingStateForAll();
        if (statusEl) statusEl.innerHTML = '<span style="color: #e62325;">Upload failed!</span>';
        throw new Error(`File upload failed with HTTP status ${response.status}. Please try again.`);
      }
    })
    .then((data) => {
      console.log("Response from /upload:", data);
      
      if (!data.success) {
        analysisInProgress = false;
        hideLoadingStateForAll();
        if (statusEl) statusEl.innerHTML = '<span style="color: #e62325;">Upload error: ' + data.msg + '</span>';
        return;
      }
      
      // Store the file path for subsequent analysis requests
      currentFilePath = data.file_path;
      
      // Begin sequential analysis process
      runSequentialAnalysis(statusEl);
    })
    .catch((error) => {
      analysisInProgress = false;
      hideLoadingStateForAll();
      console.error("Error while file uploading:", error);
      if (statusEl) statusEl.innerHTML = '<span style="color: #e62325;">Upload failed</span>';
      
      // Check if it's likely a CORS error
      let errorMessage = error.toString();
      if (error.message && error.message.includes("Failed to fetch")) {
        errorMessage = "Network error: The server may be unavailable. Please check your connection and try again.";
      }
      
      alert("Error while uploading: " + errorMessage);
    });
}

/**
 * Runs each analysis in sequence
 */
async function runSequentialAnalysis(statusEl) {
  try {
    if (!currentFilePath) {
      throw new Error("No file path available for analysis");
    }
    
    // Set status for MOST analysis
    updateSectionStatus("most", "Processing...");
    highlightSection("most");
    
    // Run MOST analysis
    const mostResult = await performAnalysis('most');
    updateAnalysisSectionWithResult("most", mostResult.msg, mostResult);
    
    // Set status for SWOT analysis
    updateSectionStatus("swot", "Processing...");
    highlightSection("swot");
    
    // Run SWOT analysis
    const swotResult = await performAnalysis('swot');
    updateAnalysisSectionWithResult("swot", swotResult.msg, swotResult);
    
    // Set status for PESTLE analysis
    updateSectionStatus("pestle", "Processing...");
    highlightSection("pestle");
    
    // Run PESTLE analysis
    const pestleResult = await performAnalysis('pestle');
    updateAnalysisSectionWithResult("pestle", pestleResult.msg, pestleResult);
    
    // Set status for Sentiment analysis
    updateSectionStatus("sentiment", "Processing...");
    highlightSection("sentiment");
    
    // Run Sentiment analysis
    const sentimentResult = await performAnalysis('sentiment');
    updateAnalysisSectionWithResult("sentiment", sentimentResult.msg, sentimentResult);
    
    // Set status for Additional analysis
    updateSectionStatus("additional", "Processing...");
    highlightSection("additional");
    
    // Determine what additional analysis is needed based on user's request
    const additionalResult = await performAnalysis('additional');
    updateAnalysisSectionWithResult("additional", additionalResult.msg, additionalResult);
    
    // Set final status
    if (statusEl) statusEl.innerHTML = '<span style="color: green;">All analyses completed successfully!</span>';
    
    // Analysis complete
    analysisInProgress = false;
    
    // Send a notification to the chatbot
    messageChatbot("All requested analyses have been completed. You can view the results above.", true);
    
  } catch (error) {
    analysisInProgress = false;
    console.error("Error during sequential analysis:", error);
    if (statusEl) statusEl.innerHTML = '<span style="color: #e62325;">Error during analysis: ' + error.message + '</span>';
    alert("Error during analysis: " + error.message);
  }
}

/**
 * Performs a single analysis by calling its API endpoint
 */
async function performAnalysis(analysisType) {
  const SERVER = window.location.origin;
  const endpoint = `/analyze/${analysisType}`;
  
  try {
    const response = await fetch(SERVER + endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        file_path: currentFilePath
      })
    });
    
    if (!response.ok) {
      throw new Error(`Analysis failed with status ${response.status}`);
    }
    
    const data = await response.json();
    if (!data.success) {
      throw new Error(data.msg || "Unknown error during analysis");
    }
    
    return data; // Return the entire data object, not just msg
  } catch (error) {
    console.error(`Error in ${analysisType} analysis:`, error);
    throw error;
  }
}

/**
 * Updates a section with the analysis result
 */
function updateAnalysisSectionWithResult(sectionType, resultText, data) {
  try {
    console.log(`Updating ${sectionType} section with result`);
    
    const targetSection = document.getElementById(sectionType);
    if (!targetSection) return;

    const contentDiv = targetSection.querySelector(".analysis-content");
    const statusElem = targetSection.querySelector(".analysis-status");
    const imagesDiv = targetSection.querySelector(".analysis-images");
    
    if (!contentDiv) return;
    
    // For extraction, we need to look for the type-specific XML tags
    const sectionTag = sectionType.toUpperCase() + "_ANALYSIS";
    const startTag = `<${sectionTag}>`;
    const endTag = `</${sectionTag}>`;
    
    let content = resultText;
    
    // Try to extract content between tags if they exist
    const startIndex = resultText.indexOf(startTag);
    const endIndex = resultText.indexOf(endTag);
    
    if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
      // Extract content between tags (excluding the tags)
      content = resultText.substring(
        startIndex + startTag.length, 
        endIndex
      ).trim();
    }
    
    // Format the content with proper spacing and line breaks
    const formattedText = content
      .replace(/\n{3,}/g, '\n\n')  // Replace excessive newlines
      .trim();
    
    // Only update text content for Additional analysis, not for MOST/SWOT/PESTLE
    if (sectionType === "additional" || sectionType === "sentiment") {
      contentDiv.innerHTML = `<pre>${formattedText}</pre>`;
    }
    
    targetSection.classList.add("updated");
    
    // Handle visualization if available (for all sections except Additional)
    if (data && data.has_visualization && sectionType !== "additional") {
      if (imagesDiv) {
        const SERVER = window.location.origin;
        
        // Determine max-width based on section type
        let maxWidth = "80%";  // Default
        if (sectionType === "most") {
          maxWidth = "100%";
        } else if (sectionType === "sentiment") {
          maxWidth = "100%";
        }
        
        // Handle single visualization path (backward compatibility)
        if (data.visualization_path) {
          const imgPath = `${SERVER}/static/${data.visualization_path}`;
          imagesDiv.innerHTML = `
            <div style="display: flex; justify-content: left; margin-bottom: 1rem;">
              <img src="${imgPath}" alt="${sectionType} visualization" 
                   style="width: auto; max-width: ${maxWidth}; height: auto; object-fit: contain; border: 1px solid #eee;" />
            </div>
          `;
        }
        // Handle multiple visualization paths
        else if (data.visualization_paths && data.visualization_paths.length > 0) {
          imagesDiv.innerHTML = '';
          data.visualization_paths.forEach(path => {
            const imgPath = `${SERVER}/static/${path}`;
            imagesDiv.innerHTML += `
              <div style="display: flex; justify-content: left; margin-bottom: 1rem;">
                <img src="${imgPath}" alt="${sectionType} visualization" 
                     style="width: auto; max-width: ${maxWidth}; height: auto; object-fit: contain; border: 1px solid #eee;" />
              </div>
            `;
          });
        }
      }
    }
    
    // Remove the updated class after animation completes
    setTimeout(() => {
      targetSection.classList.remove("updated");
    }, 2000);
    
    if (statusElem) statusElem.textContent = "Completed";
    
    // Scroll to this section
    targetSection.scrollIntoView({ behavior: "smooth", block: "start" });
    
  } catch (err) {
    console.error(`Error updating ${sectionType} result:`, err);
  }
}

/**
 * Updates the status display for a section
 */
function updateSectionStatus(sectionType, status) {
  const section = document.getElementById(sectionType);
  if (!section) return;
  
  const statusElem = section.querySelector(".analysis-status");
  if (statusElem) statusElem.textContent = status;
}

/**
 * Highlights a section to show it's currently being processed
 */
function highlightSection(sectionType) {
  // Remove highlighting from all sections
  const sections = ["most", "swot", "pestle", "sentiment", "additional"];
  sections.forEach(section => {
    const elem = document.getElementById(section);
    if (elem) elem.classList.remove("updating");
  });
  
  // Highlight the current section
  const currentSection = document.getElementById(sectionType);
  if (currentSection) {
    currentSection.classList.add("updating");
    
    // Also update the content to show it's processing
    const contentDiv = currentSection.querySelector(".analysis-content");
    if (contentDiv) {
      contentDiv.innerHTML = `
        <p class="loading-placeholder">
          <span style="display: inline-block; animation: pulse 1.5s infinite;">
            Analyzing document for ${sectionType.toUpperCase()} analysis...
          </span>
        </p>
      `;
    }
    
    // Scroll to this section
    currentSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/**
 * Updates UI to show loading state in all analysis sections
 */
function showLoadingStateForAll() {
  const sections = ["most", "swot", "pestle", "sentiment", "additional"];
  sections.forEach(section => {
    const elem = document.getElementById(section);
    if (elem) {
      const statusElem = elem.querySelector(".analysis-status");
      if (statusElem) statusElem.textContent = "Waiting...";
      
      // Clear previous content and show loading indicator
      const contentDiv = elem.querySelector(".analysis-content");
      if (contentDiv) {
        contentDiv.innerHTML = `
          <p class="loading-placeholder">
            Waiting for analysis to begin...
          </p>
        `;
      }
      
      // Also update image placeholders with the same message
      const imagePlaceholder = elem.querySelector(".analysis-images .image-placeholder");
      if (imagePlaceholder) {
        imagePlaceholder.textContent = "Waiting for analysis to begin...";
      }
    }
  });
  
  // Add pulsing animation style if not already present
  if (!document.getElementById('loading-animation-style')) {
    const style = document.createElement('style');
    style.id = 'loading-animation-style';
    style.textContent = `
      @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
      }
    `;
    document.head.appendChild(style);
  }
}

/**
 * Resets UI loading states for all analysis sections
 */
function hideLoadingStateForAll() {
  const sections = ["most", "swot", "pestle", "sentiment", "additional"];
  sections.forEach(section => {
    const elem = document.getElementById(section);
    if (elem) {
      const statusElem = elem.querySelector(".analysis-status");
      if (statusElem) statusElem.textContent = "Waiting for input";
      elem.classList.remove("updating");
      
      // Restore default content if no analysis was performed
      const contentDiv = elem.querySelector(".analysis-content");
      if (contentDiv && contentDiv.querySelector('.loading-placeholder')) {
        contentDiv.innerHTML = `<p class="loading-placeholder">Upload a file to generate ${section.toUpperCase()} analysis</p>`;
      }
    }
  });
}

/**
 * Sends a message to the Watson Assistant chatbot
 */
function messageChatbot(txt, silent = false) {
  if (!window.g_wa_instance) {
    console.error("Watson Assistant instance not initialized");
    return;
  }
  
  // Truncate text to Watson's limit
  const maxChars = 2040;
  txt = txt.substring(0, maxChars);
  
  // If text was truncated, add a note
  const wasTruncated = txt.length >= maxChars;
  if (wasTruncated) {
    txt += "\n[Note: This message was truncated to fit within message limits.]";
  }
  
  const send_obj = { input: { message_type: "text", text: txt } };

  window.g_wa_instance.send(send_obj, { silent })
    .then(() => {
      console.log("Message sent to Watson Assistant.");
    })
    .catch(function (error) {
      console.error("Sending message to chatbot failed", error);
    });
}
