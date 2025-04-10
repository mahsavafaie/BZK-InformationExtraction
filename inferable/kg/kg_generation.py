from typing import Dict, Iterable, Union
from urllib.parse import urlencode, quote
import json
import gzip
from rdflib import Literal, Namespace
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

            SCHEMA = Namespace("http://schema.org/")

            ntriples = []
            ntriples.append(f'<{base_uri}card/{image_file}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{base_uri}/BZKCard>. \n')
            ntriples.append(f'<{base_uri}/BZKCard> <https://www.w3.org/2000/01/rdf-schema#subClassOf> <https://www.ica.org/standards/RiC/ontology#Record>. \n')

            #card
            ntriples.append(f'<{base_uri}card/{image_file}> <https://www.ica.org/standards/RiC/ontology#hasCreator> <https://schema.org/Organization> . \n')
            ntriples.append(f'<{base_uri}card/{image_file}/compensationOffice> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://schema.org/Organization> . \n')
            if entry["CompensationOffice1"]:
                ntriples.append(f'<{base_uri}card/{image_file}/compensationOffice> <https://schema.org/legalName> {literal(entry["CompensationOffice1"])} . \n')
            # applicant 
            ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>. \n')
            if entry["ApplicantFirstName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://schema.org/givenName> {literal(entry["ApplicantFirstName"])} . \n')
            if entry["ApplicantLastName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://schema.org/familyName> {literal(entry["ApplicantLastName"])} . \n')
            if entry["ApplicantBirthName"] or entry["ApplicantAltFirstName"] or entry["ApplicantAltLastName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://schema.org/additionalName> {literal(entry["ApplicantBirthName"]) or literal(entry["ApplicantAltFirstName"]) or literal(entry["ApplicantAltLastName"])} . \n')
            if entry["ApplicantBirthDate"]:
                ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://schema.org/birthDate> {literal(entry["ApplicantBirthDate"])} . \n')
            if entry["ApplicantBirthPlace"]:
                ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://schema.org/birthPlace> {literal(entry["ApplicantBirthPlace"])} . \n')
            if entry["ApplicantCurrentAddress"]:
                ntriples.append(f'<{base_uri}card/{image_file}/applicant> <http://schema.org/address> {literal(entry["ApplicantCurrentAddress"])} . \n')
            
            # victim
            ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>. \n')
            if entry["VictimFirstName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://schema.org/givenName> {literal(entry["VictimFirstName"])} . \n')
            if entry["VictimLastName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://schema.org/familyName> {literal(entry["VictimLastName"])} . \n')
            if entry["VictimBirthName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://schema.org/additionalName> {literal(entry["VictimBirthName"])} . \n')
            if entry["VictimBirthName"] or entry["VictimAltFirstName"] or entry["VictimAltLastName"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://schema.org/additionalName> {literal(entry["VictimBirthName"]) or literal(entry["VictimAltFirstName"]) or literal(entry["VictimAltLastName"])} . \n')
            if entry["VictimBirthDate"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://schema.org/birthDate> {literal(entry["VictimBirthDate"])} . \n')
            if entry["VictimBirthPlace"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://schema.org/birthPlace> {literal(entry["VictimBirthPlace"])} . \n')
            if entry["VictimDeathDate"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://schema.org/deathDate> {literal(entry["VictimDeathDate"])} . \n')
            if entry["VictimDeathPlace"]:
                ntriples.append(f'<{base_uri}card/{image_file}/victim> <http://schema.org/deathPlace> {literal(entry["VictimDeathPlace"])} . \n')
            file.writelines(ntriples)
    logger.info(f"Finished writing n-triples to {out_file}")
