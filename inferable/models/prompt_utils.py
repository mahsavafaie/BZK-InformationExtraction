
# Note that the overall prompt are split in multiple lines which are concatenated during compile time
# see https://docs.python.org/3/reference/lexical_analysis.html#string-literal-concatenation
PROMPTS = {
    '0' : '<image>\nPlease provide the following information as you can see on the image as a Python dictionary. \n'
          'Use only the following keys: CompensationOffice1, BZKNr, ApplicantFirstName, ApplicantLastName, ApplicantAltFirstName, ApplicantBirthName, '
          'ApplicantAltLastName, ApplicantBirthDate, ApplicantBirthPlace, ApplicantCurrentAddress, VictimFirstName, VictimLastName, VictimAltFirstName, '
          'VictimBirthName, VictimAltLastName, VictimBirthDate, VictimBirthPlace, VictimDeathDate, VictimDeathPlace.',

    # prompt 1 includes german words for applicant, victim and BZK number  
    '1' : '<image>\nPlease provide the following information as you can see on the image as a Python dictionary. \n'
          'Use only the following keys: CompensationOffice1, BZKNr, ApplicantFirstName, ApplicantLastName, ApplicantAltFirstName, ApplicantBirthName, '
          'ApplicantAltLastName, ApplicantBirthDate, ApplicantBirthPlace, ApplicantCurrentAddress, VictimFirstName, VictimLastName, VictimAltFirstName, '
          'VictimBirthName, VictimAltLastName, VictimBirthDate, VictimBirthPlace, VictimDeathDate, VictimDeathPlace.\n'
          'When extracting the information about the applicant, look at the text below the words "Anspruchsberechtigter" or '
          '"Antragsteller". When extracting the information about the victim, look at the text below the word "Verfolgter". '
          'When extracting the information about the BZKnr, look at the text around "RegNr" or "Kartei-Nr" or "Register Nr." '
          'or "A.Z." or "Grundlisten-Nr" or "Z.K" or "Art.V-56-II-Nr" or "Eingangsnummer".',

    # prompt  includes german words for applicant, victim and BZK number + Standard date format and normalised address       
    '2' : '<image>\nPlease provide the following information as you can see on the image as a Python dictionary. \n'
          'Use only the following keys: CompensationOffice1, BZKNr, ApplicantFirstName, ApplicantLastName, ApplicantAltFirstName, ApplicantBirthName, '
          'ApplicantAltLastName, ApplicantBirthDate, ApplicantBirthPlace, ApplicantCurrentAddress, VictimFirstName, VictimLastName, VictimAltFirstName, '
          'VictimBirthName, VictimAltLastName, VictimBirthDate, VictimBirthPlace, VictimDeathDate, VictimDeathPlace.\n'
          'When extracting the information about the applicant, look at the text below the words "Anspruchsberechtigter" or '
          '"Antragsteller". When extracting the information about the victim, look at the text below the word "Verfolgter". '
          'When extracting the information about the BZKnr, look at the text around "RegNr" or "Kartei-Nr" or "Register Nr." '
          'or "A.Z." or "Grundlisten-Nr" or "Z.K" or "Art.V-56-II-Nr" or "Eingangsnummer". '
          'Convert the values for VictimDeathDate and ApplicantBirthDate and VictimBirthDate into the YYYY-MM-DD date format. '
          'From ApplicantCurrentAddress, extract the city only.',

    '3' : '<image>\nThe image is an index card from the process of Compensation for atrocities of Nazi Germany, called "Wiedergutmachung".\n'
          'Please provide the following information as you can see on the image as a Python dictionary. \n'
          'Use only the following keys: CompensationOffice1, BZKNr, ApplicantFirstName, ApplicantLastName, ApplicantAltFirstName, ApplicantBirthName, '
          'ApplicantAltLastName, ApplicantBirthDate, ApplicantBirthPlace, ApplicantCurrentAddress, VictimFirstName, VictimLastName, VictimAltFirstName, '
          'VictimBirthName, VictimAltLastName, VictimBirthDate, VictimBirthPlace, VictimDeathDate, VictimDeathPlace.\n'
          'When extracting the information about the applicant, look at the text below the words "Anspruchsberechtigter" or "Antragsteller". '
          "An applicant's death date" ' can only begin with "19". When extracting the information about the victim, look at the text below '
          'the word "Verfolgter".'" A victim's birth year can almost certainly only be before 1946. A victim's death year can only be completed"' by "19". '
          'When extracting the information about the BZKnr, look at the text around "RegNr" or "Kartei-Nr" or "Register Nr." '
          'or "A.Z." or "Grundlisten-Nr" or "Z.K" or "Art.V-56-II-Nr" or "Eingangsnummer". '
          'Convert the values for VictimDeathDate and ApplicantBirthDate and VictimBirthDate into the YYYY-MM-DD date format. '
          'From ApplicantCurrentAddress, extract the city only.',

    '4' : '<image>\nThe image is an index card from the process of Compensation for atrocities of Nazi Germany, called "Wiedergutmachung".\n'
          'Please provide the following information as you can see on the image as a Python dictionary.\n'
          'Use only the following keys: CompensationOffice1, BZKNr, ApplicantFirstName, ApplicantLastName, ApplicantAltFirstName, ApplicantBirthName, '
          'ApplicantAltLastName, ApplicantBirthDate, ApplicantBirthPlace, ApplicantCurrentAddress, VictimFirstName, VictimLastName, VictimAltFirstName, '
          'VictimBirthName, VictimAltLastName, VictimBirthDate, VictimBirthPlace, VictimDeathDate, VictimDeathPlace.\n'
          'When extracting the information about the applicant, look at the text below the words "Anspruchsberechtigter" or '
          '"Antragsteller". '"An applicant's death date"' can only begin with "19". When extracting the information about the victim, look at the text below '
          'the word "Verfolgter". '"A victim's birth year can almost certainly only be before 1946. A victim's death year can only be"' completed by "19". '
          'When extracting the information about the BZKnr, look at the text around "RegNr" or "Kartei-Nr" or "Register Nr." '
          'or "A.Z." or "Grundlisten-Nr" or "Z.K" or "Art.V-56-II-Nr" or "Eingangsnummer".',

    '5' : '<image>\nExtract BZK data.'
}

def get_prompt_id(prompt_input):
    prompt_input = str(prompt_input)
    if prompt_input in PROMPTS:
        return prompt_input
    else:
        return hash(prompt_input)

def get_prompt_text(prompt_input):
    prompt_input = str(prompt_input)
    if prompt_input in PROMPTS:
        return PROMPTS[prompt_input]
    else:
        return prompt_input