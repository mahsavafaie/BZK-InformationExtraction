from typing import Dict, Iterable, Union
from urllib.parse import urlencode, quote
import json
import gzip
from rdflib import Literal
import logging
import os

logger = logging.getLogger(__name__)

def __lazy_jsonl_reader(filename):
    open_func = gzip.open if filename.endswith('.gz') else open
    with open_func(filename, 'rt', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

def literal(literal_value: str) -> str:
    return Literal(literal_value).n3()

def write_n_triple_file(data: Union[str,Iterable[Dict[str, str]]], out_file: str, base_uri = "http://fiz.de/") -> None:
    """
    Write the given data to a file in RDF format.

    :param data: The data to write to the file either as a string (path pointing to a jsonl file) or as an iterable of dictionaries.
    :param out_file: The name of the file to write the data to.
    """
    # "file_name","CompensationOffice1","BZKNr","ApplicantFirstName","ApplicantLastName","ApplicantAltFirstName","ApplicantBirthName","ApplicantAltLastName","ApplicantBirthDate","ApplicantBirthPlace","ApplicantCurrentAddress","VictimFirstName","VictimLastName","VictimAltFirstName","VictimBirthName","VictimAltLastName","VictimBirthDate","VictimBirthPlace","VictimDeathDate","VictimDeathPlace"
    # https://stackoverflow.com/questions/40828501/how-to-encode-rdf-n-triples-string-literals
    # https://github.com/RDFLib/rdflib/issues/222
    
    data_iterable = __lazy_jsonl_reader(data) if isinstance(data, str) else data
    
    logger.info(f"Writing n-triples to {out_file}")
    # overrite the file if it exists
    with open(out_file, 'w', encoding='utf-8') as file:
        for entry in data_iterable:
            image_file = quote(os.path.join(entry['path'], entry['filename'])) 
            #image_file = quote(entry['filename'])

            ntriples = []
            ntriples.append(f'<{base_uri}card/{image_file}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{base_uri}/BZKCard>. \n')

            # applicant 
            ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>. \n')
            if entry["ApplicantFirstName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://xmlns.com/foaf/0.1/firstName> {literal(entry["ApplicantFirstName"])} . \n')
            if entry["ApplicantLastName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://xmlns.com/foaf/0.1/lastName> {literal(entry["ApplicantLastName"])} . \n')
            
            # victim
            ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>. \n')
            if entry["VictimFirstName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://xmlns.com/foaf/0.1/firstName> {literal(entry["VictimFirstName"])} . \n')
            if entry["VictimLastName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://xmlns.com/foaf/0.1/lastName> {literal(entry["VictimLastName"])} . \n')

            file.writelines(ntriples)
    logger.info(f"Finished writing n-triples to {out_file}")
