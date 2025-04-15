import fs from 'fs';
import path from 'path';
import axios from 'axios';

interface PaperReference {
  title: string;
  authors: string[];
  year: number;
  url: string;
  source?: string;
}

interface ValidationError {
  field: string;
  message: string;
}

function validateReference(reference: PaperReference): ValidationError[] {
  const errors: ValidationError[] = [];
  
  // Title validation
  if (!reference.title || reference.title.trim().length === 0) {
    errors.push({ field: 'title', message: 'Title cannot be empty' });
  }
  if (reference.title.length > 500) {
    errors.push({ field: 'title', message: 'Title is too long (max 500 characters)' });
  }
  
  // Authors validation
  if (!reference.authors || reference.authors.length === 0) {
    errors.push({ field: 'authors', message: 'At least one author is required' });
  }
  if (reference.authors.some(author => !author || author.trim().length === 0)) {
    errors.push({ field: 'authors', message: 'Author names cannot be empty' });
  }
  
  // Year validation
  if (!reference.year || reference.year < 1900 || reference.year > new Date().getFullYear()) {
    errors.push({ field: 'year', message: 'Year must be between 1900 and current year' });
  }
  
  // URL validation
  if (!reference.url || !reference.url.match(/^https?:\/\/.+/)) {
    errors.push({ field: 'url', message: 'Valid URL is required' });
  }
  if (reference.url.length > 1000) {
    errors.push({ field: 'url', message: 'URL is too long (max 1000 characters)' });
  }
  
  // Source validation
  if (!reference.source || reference.source.trim().length === 0) {
    errors.push({ field: 'source', message: 'Source is required' });
  }
  
  return errors;
}

// ... rest of the file ... 