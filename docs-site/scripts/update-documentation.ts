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

interface SemanticScholarPaper {
  paperId: string;
  title: string;
  year: number;
  authors: Array<{ name: string }>;
  url: string;
}

interface ArXivPaper {
  title: string;
  authors: string[];
  year: string;
  url: string;
}

interface DocumentationSystem {
  id: string;
  title: string;
  content: string;
  category: string;
  score: number;
  details?: {
    academicFoundations: PaperReference[];
    heuristics: any[];
    implementation: any;
  };
}

interface Documentation {
  systems: DocumentationSystem[];
  categories: any[];
}

function isDuplicate(reference: PaperReference, existingReferences: PaperReference[]): boolean {
  return existingReferences.some(existing => 
    existing.title.toLowerCase() === reference.title.toLowerCase() ||
    existing.url === reference.url
  );
}

interface SearchOptions {
  dryRun?: boolean;
  verbose?: boolean;
  databases?: string[];
}

interface SemanticScholarResponse {
  data: SemanticScholarPaper[];
}

async function searchSemanticScholar(query: string): Promise<PaperReference | null> {
  try {
    const response = await axios.get<SemanticScholarResponse>(`https://api.semanticscholar.org/graph/v1/paper/search`, {
      params: {
        query,
        limit: 1,
        fields: 'title,year,authors,url'
      }
    });

    const paper = response.data.data[0];
    if (!paper) return null;

    return {
      title: paper.title,
      authors: paper.authors.map(author => author.name),
      year: paper.year,
      url: paper.url,
      source: 'Semantic Scholar'
    };
  } catch (error) {
    console.error(`Error searching for paper "${query}":`, error);
    return null;
  }
}

async function searchArXiv(query: string): Promise<PaperReference | null> {
  try {
    const response = await axios.get<string>(`http://export.arxiv.org/api/query`, {
      params: {
        search_query: query,
        max_results: 1,
        sortBy: 'relevance',
        sortOrder: 'descending'
      }
    });

    const data = response.data;
    const titleMatch = data.match(/<title>(.*?)<\/title>/);
    const authorsMatch = data.match(/<author>(.*?)<\/author>/g);
    
    if (!titleMatch || !authorsMatch) return null;

    return {
      title: titleMatch[1],
      authors: authorsMatch.map(author => author.replace(/<[^>]+>/g, '')),
      year: new Date().getFullYear(),
      url: `https://arxiv.org/abs/${data.match(/<id>(.*?)<\/id>/)?.[1]}`,
      source: 'arXiv'
    };
  } catch (error) {
    console.error(`Error searching arXiv for "${query}":`, error);
    return null;
  }
}

async function generatePaperReferences(keywords: string[], options: SearchOptions = {}): Promise<PaperReference[]> {
  const references: PaperReference[] = [];
  
  for (const keyword of keywords) {
    const paper = await searchSemanticScholar(keyword);
    if (paper) {
      references.push(paper);
    }
    // Add delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  return references;
}

async function updateDocumentation(
  systemId: string,
  keywords: string[],
  options: SearchOptions = {}
): Promise<void> {
  const docPath = path.join(__dirname, '../public/data/documentation.json');
  const rawData = fs.readFileSync(docPath, 'utf-8');
  const documentation: Documentation = JSON.parse(rawData);

  const system = documentation.systems.find(s => s.id === systemId);
  if (!system) {
    throw new Error(`System with ID ${systemId} not found`);
  }

  // Generate new references
  const newReferences = await generatePaperReferences(keywords, options);
  
  if (options.verbose) {
    console.log(`Generated ${newReferences.length} new references`);
    newReferences.forEach(ref => {
      console.log(`- ${ref.title} (${ref.year}) from ${ref.source}`);
    });
  }

  if (options.dryRun) {
    console.log('Dry run mode - no changes will be made to documentation.json');
    return;
  }

  // Update system's academic foundations
  if (!system.details) {
    system.details = {
      academicFoundations: [],
      heuristics: [],
      implementation: { description: '', codeExamples: [] }
    };
  }
  
  // Filter out duplicates from existing references
  const existingReferences = system.details.academicFoundations;
  const uniqueNewReferences = newReferences.filter(ref => !isDuplicate(ref, existingReferences));
  
  system.details.academicFoundations = [
    ...existingReferences,
    ...uniqueNewReferences
  ];

  // Write updated documentation back to file
  fs.writeFileSync(
    docPath,
    JSON.stringify(documentation, null, 2),
    'utf-8'
  );

  console.log(`Successfully updated documentation for system ${systemId}`);
  console.log(`Added ${uniqueNewReferences.length} new references`);
  if (newReferences.length !== uniqueNewReferences.length) {
    console.log(`Skipped ${newReferences.length - uniqueNewReferences.length} duplicate references`);
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.error('Usage: ts-node update-documentation.ts <system-id> <keyword1> [keyword2 ...] [options]');
    console.error('Options:');
    console.error('  --dry-run     Preview changes without saving');
    console.error('  --verbose     Show detailed output');
    console.error('  --databases   Comma-separated list of databases (semantic-scholar,arxiv)');
    process.exit(1);
  }

  const options: SearchOptions = {
    dryRun: args.includes('--dry-run'),
    verbose: args.includes('--verbose'),
    databases: args.find(arg => arg.startsWith('--databases='))?.split('=')[1]?.split(',')
  };

  const positionalArgs = args.filter(arg => !arg.startsWith('--'));
  const [systemId, ...keywords] = positionalArgs;
  
  try {
    await updateDocumentation(systemId, keywords, options);
  } catch (error) {
    console.error('Error updating documentation:', error);
    process.exit(1);
  }
}

main(); 