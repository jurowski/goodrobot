import fs from 'fs';
import path from 'path';

interface AcademicPaper {
  title: string;
  year: number;
  authors: string[];
  source: string;
  doi: string;
  url: string;
}

interface Link {
  title: string;
  url: string;
  type: 'wikipedia' | 'academic' | 'tutorial';
}

interface Heuristic {
  title: string;
  description: string;
  links: Link[];
  relationships?: Array<{
    heuristic: string;
    description: string;
    system: string;
  }>;
}

interface SystemDetails {
  academicFoundations: AcademicPaper[];
  heuristics: Heuristic[];
  implementation: {
    description: string;
    codeExamples: Array<{
      title: string;
      language: string;
      code: string;
    }>;
  };
}

interface DocumentationSystem {
  id: string;
  title: string;
  content: string;
  category: string;
  score: number;
  tags?: string[];
  lastUpdated?: string;
  details?: SystemDetails;
}

interface Documentation {
  systems: DocumentationSystem[];
  categories: Array<{
    id: string;
    name: string;
    description: string;
  }>;
}

// Educational resources mapping
const EDUCATIONAL_RESOURCES: Record<string, Link[]> = {
  'pattern analysis': [
    {
      title: 'Pattern Recognition',
      url: 'https://en.wikipedia.org/wiki/Pattern_recognition',
      type: 'wikipedia'
    },
    {
      title: 'Machine Learning Patterns',
      url: 'https://www.coursera.org/learn/machine-learning',
      type: 'tutorial'
    }
  ],
  'state management': [
    {
      title: 'State Management',
      url: 'https://en.wikipedia.org/wiki/State_management',
      type: 'wikipedia'
    },
    {
      title: 'Distributed State',
      url: 'https://www.edx.org/course/distributed-systems',
      type: 'tutorial'
    }
  ],
  // Add more mappings as needed
};

// Relationship mapping between systems
const SYSTEM_RELATIONSHIPS: Record<string, Array<{
  system: string;
  heuristic: string;
  description: string;
}>> = {
  'lock-migration': [
    {
      system: 'lock-transformation',
      heuristic: 'State Adaptation',
      description: 'Migration patterns inform transformation strategies'
    }
  ],
  // Add more relationships as needed
};

function enhanceHeuristic(heuristic: Heuristic, systemId: string): Heuristic {
  // Add educational links based on heuristic title and description
  const keywords = [...heuristic.title.toLowerCase().split(' '), ...heuristic.description.toLowerCase().split(' ')];
  const links = new Set<Link>();
  
  keywords.forEach(keyword => {
    if (EDUCATIONAL_RESOURCES[keyword]) {
      EDUCATIONAL_RESOURCES[keyword].forEach(link => links.add(link));
    }
  });

  // Add relationships if they exist
  const relationships = SYSTEM_RELATIONSHIPS[systemId]?.filter(
    rel => rel.heuristic.toLowerCase().includes(heuristic.title.toLowerCase())
  );

  return {
    ...heuristic,
    links: [...heuristic.links, ...Array.from(links)],
    relationships: relationships || []
  };
}

function processDocumentation(doc: Documentation): Documentation {
  return {
    ...doc,
    systems: doc.systems.map(system => ({
      ...system,
      details: system.details ? {
        ...system.details,
        heuristics: system.details.heuristics.map(heuristic => 
          enhanceHeuristic(heuristic, system.id)
        )
      } : undefined
    }))
  };
}

function validateAcademicPaper(paper: AcademicPaper): string[] {
  const errors: string[] = [];
  if (!paper.title) errors.push('Missing title');
  if (!paper.year || paper.year < 1900 || paper.year > new Date().getFullYear()) errors.push('Invalid year');
  if (!paper.authors?.length) errors.push('Missing authors');
  if (!paper.source) errors.push('Missing source');
  if (!paper.doi) errors.push('Missing DOI');
  if (!paper.url) errors.push('Missing URL');
  return errors;
}

function validateHeuristic(heuristic: Heuristic): string[] {
  const errors: string[] = [];
  if (!heuristic.title) errors.push('Missing title');
  if (!heuristic.description) errors.push('Missing description');
  if (!heuristic.links?.length) errors.push('Missing links');
  return errors;
}

function validateSystem(system: DocumentationSystem): string[] {
  const errors: string[] = [];
  if (!system.id) errors.push('Missing system ID');
  if (!system.title) errors.push('Missing title');
  if (!system.content) errors.push('Missing content');
  if (!system.category) errors.push('Missing category');
  if (system.score < 0 || system.score > 1) errors.push('Invalid score');
  
  if (system.details) {
    system.details.academicFoundations.forEach((paper, index) => {
      const paperErrors = validateAcademicPaper(paper);
      if (paperErrors.length) {
        errors.push(`Academic paper ${index + 1} errors: ${paperErrors.join(', ')}`);
      }
    });
    
    system.details.heuristics.forEach((heuristic, index) => {
      const heuristicErrors = validateHeuristic(heuristic);
      if (heuristicErrors.length) {
        errors.push(`Heuristic ${index + 1} errors: ${heuristicErrors.join(', ')}`);
      }
    });
  }
  
  return errors;
}

function validateDocumentation(doc: Documentation): { isValid: boolean; errors: string[] } {
  const errors: string[] = [];
  
  if (!doc.systems?.length) {
    errors.push('No systems found');
  } else {
    doc.systems.forEach((system, index) => {
      const systemErrors = validateSystem(system);
      if (systemErrors.length) {
        errors.push(`System ${index + 1} (${system.id}) errors: ${systemErrors.join(', ')}`);
      }
    });
  }
  
  if (!doc.categories?.length) {
    errors.push('No categories found');
  } else {
    doc.categories.forEach((category, index) => {
      if (!category.id) errors.push(`Category ${index + 1} missing ID`);
      if (!category.name) errors.push(`Category ${index + 1} missing name`);
      if (!category.description) errors.push(`Category ${index + 1} missing description`);
    });
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
}

// Main execution
const inputPath = path.join(__dirname, '../public/data/documentation.json');
const outputPath = path.join(__dirname, '../public/data/documentation-enhanced.json');

try {
  const rawData = fs.readFileSync(inputPath, 'utf-8');
  const documentation: Documentation = JSON.parse(rawData);
  
  // Validate the documentation
  const validation = validateDocumentation(documentation);
  if (!validation.isValid) {
    console.error('Documentation validation failed:');
    validation.errors.forEach(error => console.error(`- ${error}`));
    process.exit(1);
  }
  
  const enhancedDocumentation = processDocumentation(documentation);
  
  // Validate the enhanced documentation
  const enhancedValidation = validateDocumentation(enhancedDocumentation);
  if (!enhancedValidation.isValid) {
    console.error('Enhanced documentation validation failed:');
    enhancedValidation.errors.forEach(error => console.error(`- ${error}`));
    process.exit(1);
  }
  
  fs.writeFileSync(
    outputPath,
    JSON.stringify(enhancedDocumentation, null, 2),
    'utf-8'
  );
  
  console.log('Documentation processed and validated successfully!');
  console.log(`Enhanced documentation written to: ${outputPath}`);
} catch (error) {
  console.error('Error processing documentation:', error);
  process.exit(1);
} 