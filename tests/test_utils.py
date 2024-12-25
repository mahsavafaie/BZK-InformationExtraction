from inferable.models.utils import extract_xml_info, extract_json_info, align_keys

# test xml info extraction

def test_xml_extraction():
    text = "<s_BZKNr>bla<s_BZKNr>8307/IV/311</s_BZKNr>blub</s_BZKNr>"
    extract_xml_info(text, "BZKNr", smallest_distance=True)

    assert(extract_xml_info(text, "BZKNr", smallest_distance=True) == "8307/IV/311")
    assert(extract_xml_info(text, "BZKNr", smallest_distance=False, remove_tags_inside=False) == "bla<s_BZKNr>8307/IV/311</s_BZKNr>blub")
    assert(extract_xml_info(text, "BZKNr", smallest_distance=False, remove_tags_inside=True) == "bla8307/IV/311blub")

def test_partial_xml_extraction():
    text = "<s_BZKNr>abc</s_bar>"
    assert(extract_xml_info(text, "BZKNr", smallest_distance=True, allow_partial_match=False) == "")
    assert(extract_xml_info(text, "BZKNr", smallest_distance=True, allow_partial_match=True) == "abc")

    text = ">abc</s_BZKNr>"
    assert(extract_xml_info(text, "BZKNr", smallest_distance=True, allow_partial_match=False) == "")
    assert(extract_xml_info(text, "BZKNr", smallest_distance=True, allow_partial_match=True) == "abc")

    text = "abc</s_BZKNr>"
    assert(extract_xml_info(text, "BZKNr", smallest_distance=True, allow_partial_match=False) == "")
    assert(extract_xml_info(text, "BZKNr", smallest_distance=True, allow_partial_match=True) == "abc")

    text = "<s_BZKNr>abc"
    assert(extract_xml_info(text, "BZKNr", smallest_distance=True, allow_partial_match=False) == "")
    assert(extract_xml_info(text, "BZKNr", smallest_distance=True, allow_partial_match=True) == "abc")

def test_partial_xml_extraction_two():
    text ="<s_BZKNr> 2/ 01884</s_BZKNr><s_ApplicantFirstName> Josef</s_ApplicantCurrentAddress>"
    assert(extract_xml_info(text, "ApplicantFirstName", allow_partial_match=True) == "Josef")



# test json info extraction

def test_json():
    text = """```python
{
    "Applicant First Name": "Max",
    "Applicant Last Name": "Muster",
    "Applicant Birthdate": "10.5.1913",
    "Applicant Birthplace": "Paris",
    "Applicant Address": "Paris           ",
    "Victim First Name": "Hartuf
    "Victim Last Name": foo Bar,
    "Victim Birthdate": None,
    "Victim Birthplace": "Nathania   "136",
    "BZK number": "431  "387",
    "Marital Status": null,
    "Victim Death Date": "null", "Victim Death Place": "null"
}
```"""
    gt = {
        "Applicant First Name": "Max",
        "Applicant Last Name": "Muster",
        "Applicant Birthdate": "10.5.1913",
        "Applicant Birthplace": "Paris",
        "Applicant Address": "Paris",
        "Victim First Name": "Hartuf",
        "Victim Last Name": "foo Bar",
        "Victim Birthdate": None,
        "Victim Birthplace": "Nathania 136",
        "BZK number": "431 387",
        "Marital Status": None,
        "Victim Death Date": None,
        "Victim Death Place": None
    }

    assert(gt == extract_json_info(text))


def test_json_two():
    """Test json extraction with a text that never closed correctly"""

    text = """```python
{
    "Applicant First Name": "Max",
    "Applicant Last Name": "Mustermann",
    "Applicant Birthdate": null,
    "Applicant Birthplace": null,
    "Applicant Address": "Houston, Texas Hamilton                   ,   ,   ,   ,  """

    gt = {
        "Applicant First Name": "Max",
        "Applicant Last Name": "Mustermann",
        "Applicant Birthdate": None,
        "Applicant Birthplace": None,
        "Applicant Address": "Houston, Texas Hamilton",
    }

    assert(gt == extract_json_info(text))

def test_json_one_line():

    text = """```python
{
    Applicant First Name: Max, "Applicant Last Name": Mustermann, Applicant Birthdate:2024-01-01,Applicant Address:null"""
    
    gt = {
        "Applicant First Name": "Max",
        "Applicant Last Name": "Mustermann",
        "Applicant Birthdate": "2024-01-01",
        "Applicant Address": None,
    }

    assert(gt == extract_json_info(text))

def test_json_wrong_keys():
    """Test json extraction where the keys are not correct"""

    text = """```python
{
    Applicant First Name: Max,
    Applicant Last Name: Mustermann,
    "Applicant Birthdate": null}"""
    
    gt = {
        "Applicant First Name": "Max",
        "Applicant Last Name": "Mustermann",
        "Applicant Birthdate": None,
    }

    assert(gt == extract_json_info(text))


def test_json_where_values_contains_comma():
    """Test json extraction where the keys are not correct"""

    text = """```python
{
    "Applicant First Name": "Max, Muster:mann, Foo,",
    Applicant Last Name: ",Mustermann, ",
    "Applicant Birthdate": null}"""
    
    gt = {
        "Applicant First Name": "Max, Muster:mann, Foo",
        "Applicant Last Name": "Mustermann",
        "Applicant Birthdate": None,
    }

    assert(gt == extract_json_info(text))

def test_valid_json():
    """Test json extraction where the json is completely correct"""

    text = """```python
{
    "Applicant First Name": "Max", "Applicant Last Name": "Mustermann", "Applicant Birthdate":"2024-01-01", "Applicant Address":null }"""
    
    gt = {
        "Applicant First Name": "Max",
        "Applicant Last Name": "Mustermann",
        "Applicant Birthdate": "2024-01-01",
        "Applicant Address": None,
    }

    assert(gt == extract_json_info(text))

#### Test the align_keys function

def test_align_keys():
    ground_truth_keys = ["Applicant First Name", "Applicant Last Name", "Applicant Birthdate", "Applicant Birthplace", "Applicant Address"]
    extracted_dict = {
        "Applicant First Name": "Max",# correct
        "Applicant_last_name": "Muster", # undersores
        "Applicant Birthhdate": "10.5.1913", # added h
        "Birthplace": "Paris", # missing Applicant
        "Applicant Addressss": "Paris" # added more s
    }

    aligned_dict = {
        "Applicant First Name": "Max",
        "Applicant Last Name": "Muster",
        "Applicant Birthdate": "10.5.1913",
        "Applicant Birthplace": "Paris",
        "Applicant Address": "Paris",
    }

    assert(aligned_dict == align_keys(ground_truth_keys, extracted_dict))
