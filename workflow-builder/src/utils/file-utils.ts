/**
 * Utility functions for file operations such as saving/loading files
 */

/**
 * Save data to a file and trigger download
 * @param data The content to save
 * @param filename Suggested filename for the download
 * @param type The MIME type of the file
 */
export const saveToFile = (data: string, filename: string, type: string = 'text/plain'): void => {
  // Create a blob with the data
  const blob = new Blob([data], { type });
  
  // Create a URL for the blob
  const url = URL.createObjectURL(blob);
  
  // Create a temporary link element
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  
  // Append to the document
  document.body.appendChild(link);
  
  // Programmatically click the link to trigger the download
  link.click();
  
  // Clean up
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Read a file as text
 * @param file The file object to read
 * @returns A promise that resolves to the file contents
 */
export const readFileAsText = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (event) => {
      if (event.target?.result) {
        resolve(event.target.result as string);
      } else {
        reject(new Error('Failed to read file'));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Error reading file'));
    };
    
    reader.readAsText(file);
  });
};

/**
 * Create a file input, let user select one or more files, and return the files
 * @param accept File types to accept, e.g. '.yaml,.yml'
 * @param multiple Whether to allow multiple file selection
 * @returns A promise that resolves to an array of selected File objects
 */
export const openFileDialog = (accept: string = '.yaml,.yml', multiple: boolean = false): Promise<File[]> => {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = accept;
    
    if (multiple) {
      input.multiple = true;
    }
    
    input.onchange = (event) => {
      const target = event.target as HTMLInputElement;
      const files = target.files;
      
      if (!files || files.length === 0) {
        reject(new Error('No file selected'));
        return;
      }
      
      // Convert FileList to Array
      const fileArray = Array.from(files);
      resolve(fileArray);
    };
    
    // Trigger the file selection dialog
    input.click();
  });
};