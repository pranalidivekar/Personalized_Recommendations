# Tailoring recommendations

## Description
A personalized recommendation system that focuses primarily on addressing the cold start problem and the latency problem. This system reduces the latency of recommendation generation from 1 minute to 5 seconds and achieves a relevancy rate of 68%.

## Data Sources
YouTube public API

## Project Structure
/project-directory
|-- /data
|-- /docs
|-- /src
|-- /tests
|-- /LICENSE
|-- README.md
|-- requirements.txt

- `data`: Directory to store data files.
- `docs`: Directory for documentation.
- `src`: Directory for Python scripts and source code.
- `tests`: Directory for implementing the system.


## Installation
1. Clone this repository to your local machine.
2. Install the required Python packages by running `pip install -r requirements.txt` in your terminal.

## Testing
- Run the test.py to get near-live data from the public API.
- Run the recommendation_final.py to fetch results
- Run the recommend_without_cache.py to see the difference in the recommendation generation latency with and without cache usage.

## Contributing
Please feel free to fork this repository, make changes, and submit pull requests. Any contributions, no matter how small, are greatly appreciated.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact
If you have any questions or feedback, please feel free to contact me at pranalidivekar@gmail.com.

