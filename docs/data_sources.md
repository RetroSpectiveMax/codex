# Data Source Reference

The production version of the Car Reliability Prediction Engine should integrate the following sources:

## National Highway Traffic Safety Administration (NHTSA)
- Endpoint: https://www.nhtsa.gov/vehicle/complaints
- Access: Bulk downloads are provided via CSV. Confirm the most recent schema before ingestion.
- Notes: Includes complaint narratives, component categories, and incident dates that can extend the NLP module.

## Consumer Reports Reliability & Ownership Data
- Endpoint: https://www.consumerreports.org/cars/
- Access: Requires a paid subscription. Scrape responsibly or negotiate API/data-sharing agreements.
- Notes: Incorporate predicted annual repair costs and expert ratings to calibrate the cost-of-ownership module.

## Owner Forums & Community Sites
- [https://www.carcomplaints.com/](https://www.carcomplaints.com/)
- [https://www.reddit.com/r/Cartalk/](https://www.reddit.com/r/Cartalk/)
- Brand-specific forums such as [https://www.teslamotorsclub.com/](https://www.teslamotorsclub.com/) or [https://www.bimmerfest.com/forums/](https://www.bimmerfest.com/forums/)

Before automating collection:
1. Review each site's Terms of Service for scraping allowances.
2. Identify available RSS feeds, APIs, or structured exports to reduce scraping overhead.
3. Establish polite crawl policies (user-agent headers, throttling) even when no rate limits are published.

## Suggested Integration Workflow
1. Use the provided synthetic dataset to prototype pipelines.
2. Develop extractors for each source and validate schema mappings in isolated notebooks.
3. Merge datasets using make/model/year keys and align complaint taxonomies.
4. Continuously evaluate model drift as fresh complaints arrive.
