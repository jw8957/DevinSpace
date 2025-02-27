# Chinese Research Queries

This project implements the methodology described in the paper "Researchy Questions: A Dataset of Multi-Perspective, Decompositional Questions for LLM Web Agents" to generate Chinese research queries.

## Implementation

The implementation follows the five-stage process described in the paper:

1. **Stage 1: Mining Search Logs** - Using Zhihu questions as source data
2. **Stage 2: Factoid Classifier** - Filtering out factoid questions with simple, direct answers
3. **Stage 3: Decompositional Classifier** - Ensuring questions can be broken down into sub-questions
4. **Stage 4: Deduplication** - Removing duplicate or very similar questions
5. **Stage 5: Final Filtering** - Applying quality filters to ensure all questions meet criteria

## Files

- `enhanced_zhihu_scraper.py`: Script to generate Zhihu-like questions across different topics
- `enhanced_query_pipeline.py`: Script implementing the five-stage pipeline
- `enhanced_zhihu_data/`: Directory containing the generated Zhihu questions
- `enhanced_pipeline_output/`: Directory containing the output of each stage of the pipeline

## Generated Queries

The implementation generates 500 Chinese research queries with the following characteristics:

- Coverage across 84 different topics
- Complexity distribution: ~51% level 2, ~49% level 3
- Persona distribution: ~31% with persona, ~69% without persona
- Diverse query styles with various structures and formats

## Usage

To generate the research queries:

1. Run the enhanced Zhihu scraper:
   ```
   python enhanced_zhihu_scraper.py
   ```

2. Process the questions through the pipeline:
   ```
   python enhanced_query_pipeline.py
   ```

The final research queries will be saved to `enhanced_pipeline_output/final_research_queries.json`.
