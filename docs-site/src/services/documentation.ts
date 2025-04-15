export interface AcademicPaper {
  title: string;
  year: number;
  authors: string[];
  source: string;
  doi: string;
  url: string;
}

export interface CodeExample {
  title: string;
  language: string;
  code: string;
}

export interface Implementation {
  description: string;
  codeExamples: CodeExample[];
}

export interface Category {
  id: string;
  name: string;
  description: string;
}

export interface Heuristic {
  title: string;
  description: string;
  links: {
    title: string;
    url: string;
    type?: 'wikipedia' | 'academic' | 'tutorial';
  }[];
  relationships?: Array<{
    heuristic: string;
    description: string;
    system: string;
  }>;
}

export interface SystemDetails {
  academicFoundations: AcademicPaper[];
  heuristics: Heuristic[];
  implementation: Implementation;
}

export interface DocumentationSystem {
  id: string;
  title: string;
  content: string;
  category: string;
  score: number;
  tags?: string[];
  lastUpdated?: string;
  details?: SystemDetails;
}

export interface Documentation {
  systems: DocumentationSystem[];
  categories: Category[];
}

export async function loadDocumentation(): Promise<Documentation> {
  const response = await fetch('/data/documentation.json');
  if (!response.ok) {
    throw new Error('Failed to load documentation');
  }
  return response.json();
}
