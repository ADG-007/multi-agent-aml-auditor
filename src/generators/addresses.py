"""Customer addresses in each country's real format. Cosmetic: no stage parses an address.

Indian localities carry their real PIN codes. Foreign addresses follow local conventions.
House numbers and streets are invented.
"""

from __future__ import annotations

import random

#: City, state, and localities with their real India Post PIN codes. The cities are weighted
#: toward where retail bank customers actually cluster: the metros carry more localities.
INDIAN_LOCALITIES: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("Mumbai", "Maharashtra", (
        ("Andheri West", "400058"), ("Bandra West", "400050"), ("Powai", "400076"),
        ("Dadar West", "400028"), ("Borivali West", "400092"), ("Malad West", "400064"),
        ("Chembur", "400071"), ("Ghatkopar East", "400077"), ("Mulund West", "400080"),
        ("Goregaon East", "400063"),
    )),
    ("New Delhi", "Delhi", (
        ("Lajpat Nagar", "110024"), ("Karol Bagh", "110005"), ("Janakpuri", "110058"),
        ("Saket", "110017"), ("Pitampura", "110034"), ("Vasant Kunj", "110070"),
        ("Greater Kailash", "110048"), ("Mayur Vihar Phase 1", "110091"),
        ("Dwarka Sector 6", "110075"), ("Rohini Sector 7", "110085"),
    )),
    ("Bengaluru", "Karnataka", (
        ("Koramangala", "560034"), ("Indiranagar", "560038"), ("Jayanagar", "560011"),
        ("Whitefield", "560066"), ("Malleshwaram", "560003"), ("HSR Layout", "560102"),
        ("BTM Layout", "560076"), ("Rajajinagar", "560010"), ("Banashankari", "560050"),
    )),
    ("Chennai", "Tamil Nadu", (
        ("Adyar", "600020"), ("T. Nagar", "600017"), ("Anna Nagar", "600040"),
        ("Velachery", "600042"), ("Mylapore", "600004"), ("Nungambakkam", "600034"),
        ("Tambaram", "600045"),
    )),
    ("Hyderabad", "Telangana", (
        ("Banjara Hills", "500034"), ("Jubilee Hills", "500033"), ("Kukatpally", "500072"),
        ("Madhapur", "500081"), ("Ameerpet", "500016"), ("Dilsukhnagar", "500060"),
        ("Gachibowli", "500032"),
    )),
    ("Kolkata", "West Bengal", (
        ("Ballygunge", "700019"), ("Behala", "700034"), ("Tollygunge", "700033"),
        ("Dum Dum", "700028"), ("Salt Lake", "700064"), ("Park Circus", "700017"),
    )),
    ("Pune", "Maharashtra", (
        ("Kothrud", "411038"), ("Aundh", "411007"), ("Hadapsar", "411028"),
        ("Viman Nagar", "411014"), ("Baner", "411045"), ("Wakad", "411057"),
        ("Shivajinagar", "411005"),
    )),
    ("Ahmedabad", "Gujarat", (
        ("Navrangpura", "380009"), ("Satellite", "380015"), ("Vastrapur", "380015"),
        ("Maninagar", "380008"), ("Paldi", "380007"), ("Bopal", "380058"),
    )),
    ("Jaipur", "Rajasthan", (
        ("Malviya Nagar", "302017"), ("Vaishali Nagar", "302021"), ("Mansarovar", "302020"),
        ("C-Scheme", "302001"), ("Raja Park", "302004"),
    )),
    ("Lucknow", "Uttar Pradesh", (
        ("Gomti Nagar", "226010"), ("Hazratganj", "226001"), ("Aliganj", "226024"),
        ("Indira Nagar", "226016"),
    )),
    ("Chandigarh", "Chandigarh", (
        ("Sector 17", "160017"), ("Sector 22", "160022"), ("Sector 8", "160009"),
        ("Sector 35", "160022"),
    )),
    ("Kochi", "Kerala", (
        ("Kakkanad", "682030"), ("Edappally", "682024"), ("Kaloor", "682017"),
        ("Vyttila", "682019"), ("Panampilly Nagar", "682036"),
    )),
    ("Thiruvananthapuram", "Kerala", (
        ("Pattom", "695004"), ("Kowdiar", "695003"), ("Vazhuthacaud", "695014"),
    )),
    ("Ludhiana", "Punjab", (
        ("Sarabha Nagar", "141001"), ("Model Town", "141002"), ("BRS Nagar", "141012"),
    )),
    ("Indore", "Madhya Pradesh", (("Vijay Nagar", "452010"), ("Palasia", "452001"))),
    ("Bhopal", "Madhya Pradesh", (("Arera Colony", "462016"), ("MP Nagar", "462011"))),
    ("Patna", "Bihar", (
        ("Kankarbagh", "800020"), ("Rajendra Nagar", "800016"), ("Boring Road", "800001"),
    )),
    ("Nagpur", "Maharashtra", (("Dharampeth", "440010"), ("Sitabuldi", "440012"))),
    ("Coimbatore", "Tamil Nadu", (
        ("RS Puram", "641002"), ("Gandhipuram", "641012"), ("Peelamedu", "641004"),
    )),
    ("Bhubaneswar", "Odisha", (
        ("Saheed Nagar", "751007"), ("Nayapalli", "751012"), ("Patia", "751024"),
    )),
    ("Guwahati", "Assam", (
        ("Dispur", "781006"), ("Paltan Bazaar", "781008"), ("Beltola", "781028"),
    )),
    ("Visakhapatnam", "Andhra Pradesh", (
        ("MVP Colony", "530017"), ("Dwaraka Nagar", "530016"), ("Gajuwaka", "530026"),
    )),
    ("Surat", "Gujarat", (("Adajan", "395009"), ("Athwa Lines", "395007"), ("Vesu", "395007"))),
)

#: Road names of the kind found in almost every Indian city.
INDIAN_ROADS = (
    "MG Road", "Station Road", "Link Road", "Main Road", "Temple Street", "Church Road",
    "Nehru Road", "Gandhi Road", "Tilak Road", "Sardar Patel Road", "Ambedkar Road",
    "Market Road", "Hospital Road", "Park Road", "College Road", "Lake Road", "Ring Road",
    "Netaji Subhash Road", "Vivekananda Road", "Shivaji Road",
)

#: Residential building and society names in the common Indian style.
INDIAN_BUILDINGS = (
    "Sai Krupa Apartments", "Shanti Kunj", "Gokul Dham Society", "Sunshine Residency",
    "Krishna Towers", "Lakshmi Nivas", "Om Sai Apartments", "Green Park Society",
    "Royal Enclave", "Silver Oak Residency", "Vrindavan Heights", "Srinivasa Nilayam",
    "Ganga Apartments", "Anand Vihar Society", "Prestige Residency", "Mangal Murti Apartments",
)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def indian_address(rng: random.Random) -> str:
    """House, building, road, locality, city, state and a PIN code that matches them."""
    city, state, localities = rng.choice(INDIAN_LOCALITIES)
    locality, pin = rng.choice(localities)

    if rng.random() < 0.6:
        # A flat in a named building - the commonest urban address.
        house = f"Flat {rng.randint(1, 14)}0{rng.randint(1, 8)}"
        building = rng.choice(INDIAN_BUILDINGS)
        head = f"{house}, {building}"
    else:
        # An independent house, which has a house number and no building name.
        head = f"H.No. {rng.randint(1, 999)}"

    # Bengaluru addresses are conventionally given by cross and main road numbers.
    if city == "Bengaluru" and rng.random() < 0.7:
        road = f"{_ordinal(rng.randint(1, 18))} Cross, {_ordinal(rng.randint(1, 12))} Main"
    else:
        road = rng.choice(INDIAN_ROADS)

    return f"{head}, {road}, {locality}, {city}, {state} {pin}"


# --- Foreign addresses ---------------------------------------------------------------------
# One builder per NRI country, in areas with large Indian diaspora populations.

_US_STREETS = ("Oak Street", "Maple Avenue", "Park Avenue", "Washington Street", "Elm Street",
               "Cedar Lane", "Lakeview Drive", "Hillcrest Road", "Sunset Boulevard", "Main Street")
#: (city, state, real ZIP codes)
_US_CITIES = (
    ("Edison", "NJ", ("08817", "08820", "08837")), ("Jersey City", "NJ", ("07302", "07306")),
    ("Sunnyvale", "CA", ("94085", "94086", "94087")), ("Fremont", "CA", ("94536", "94538", "94539")),
    ("Plano", "TX", ("75024", "75025", "75093")), ("Irving", "TX", ("75038", "75039", "75063")),
    ("Redmond", "WA", ("98052",)), ("Chicago", "IL", ("60659", "60645")),
)


def _us(rng: random.Random) -> str:
    city, state, zips = rng.choice(_US_CITIES)
    apartment = f", Apt {rng.randint(1, 30)}{rng.choice('ABCD')}" if rng.random() < 0.5 else ""
    return f"{rng.randint(100, 9999)} {rng.choice(_US_STREETS)}{apartment}, {city}, {state} {rng.choice(zips)}"


#: (emirate, areas). The UAE has no postcode system; addresses run by building and area.
_UAE_AREAS = (
    ("Dubai", ("Al Barsha", "Deira", "Bur Dubai", "Al Karama", "Al Nahda", "Jumeirah Lake Towers",
               "International City")),
    ("Abu Dhabi", ("Al Khalidiyah", "Mussafah", "Al Reem Island")),
    ("Sharjah", ("Al Majaz", "Al Taawun", "Al Nahda")),
)


def _uae(rng: random.Random) -> str:
    emirate, areas = rng.choice(_UAE_AREAS)
    area = rng.choice(areas)
    building = f"{rng.choice(('Al Noor', 'Al Fattan', 'Al Waleed', 'Rose', 'Emerald', 'Palm'))} Tower"
    return f"Apartment {rng.randint(1, 25)}{rng.randint(0, 1)}{rng.randint(1, 9)}, {building}, {area}, {emirate}"


#: (city, district, postal code). Codes sit in each city's real range, pairing not verified.
_SAUDI_DISTRICTS = (
    ("Riyadh", "Al Olaya", "12211"), ("Riyadh", "Al Malaz", "12831"), ("Riyadh", "Al Sulimaniyah", "12245"),
    ("Jeddah", "Al Rawdah", "23434"), ("Jeddah", "Al Aziziyah", "23334"),
    ("Dammam", "Al Faisaliyah", "32272"), ("Al Khobar", "Al Aqrabiyah", "34446"),
)


def _saudi(rng: random.Random) -> str:
    city, district, postcode = rng.choice(_SAUDI_DISTRICTS)
    street = rng.choice(("King Fahd Road", "Prince Sultan Street", "Al Tahlia Street", "King Abdulaziz Road"))
    return f"Building {rng.randint(1000, 9999)}, {street}, {district} District, {city} {postcode}"


#: (town, real outward postcode districts)
_UK_AREAS = (
    ("Southall", ("UB1", "UB2")), ("Wembley", ("HA0", "HA9")), ("Harrow", ("HA1", "HA2")),
    ("East Ham, London", ("E6",)), ("Ilford", ("IG1", "IG3")), ("Hounslow", ("TW3", "TW4")),
    ("Leicester", ("LE4", "LE5")), ("Birmingham", ("B20", "B21")), ("Slough", ("SL1", "SL2")),
)
_UK_STREETS = ("Green Street", "Station Road", "Church Lane", "Park Avenue", "Kingsway",
               "Victoria Road", "Queens Road", "The Broadway", "High Street")
#: Letters never used in the inward part of a UK postcode.
_UK_INWARD_LETTERS = "ABDEFGHJLNPQRSTUWXYZ"


def _uk(rng: random.Random) -> str:
    town, districts = rng.choice(_UK_AREAS)
    inward = f"{rng.randint(1, 9)}{rng.choice(_UK_INWARD_LETTERS)}{rng.choice(_UK_INWARD_LETTERS)}"
    flat = f"Flat {rng.randint(1, 12)}, " if rng.random() < 0.4 else ""
    return f"{flat}{rng.randint(1, 220)} {rng.choice(_UK_STREETS)}, {town} {rng.choice(districts)} {inward}"


#: (HDB town, the two-digit postal sector its blocks sit in)
_SG_TOWNS = (("Tampines", "52"), ("Jurong West", "64"), ("Woodlands", "73"), ("Ang Mo Kio", "56"),
             ("Bedok", "46"), ("Sengkang", "54"), ("Serangoon", "55"))


def _singapore(rng: random.Random) -> str:
    # HDB format: the six-digit postcode is the sector followed by the block number.
    town, sector = rng.choice(_SG_TOWNS)
    block = rng.randint(101, 899)
    return (f"Blk {block} {town} Street {rng.randint(1, 9)}{rng.randint(1, 4)}, "
            f"#{rng.randint(2, 16):02d}-{rng.randint(10, 450)}, Singapore {sector}0{block}")


#: (city, province, real forward sortation areas)
_CA_AREAS = (
    ("Brampton", "ON", ("L6P", "L6R", "L6Y", "L6Z")), ("Mississauga", "ON", ("L5B", "L5M")),
    ("Toronto", "ON", ("M1B", "M1S")), ("Surrey", "BC", ("V3W", "V3S")), ("Calgary", "AB", ("T3J",)),
)
_CA_STREETS = ("Sandalwood Parkway", "Queen Street", "Dixie Road", "Maple Drive", "King Street",
               "Bovaird Drive", "Scott Road")
#: Letters Canada Post never uses in a postcode.
_CA_LETTERS = "ABCEGHJKLMNPRSTVWXYZ"


def _canada(rng: random.Random) -> str:
    city, province, prefixes = rng.choice(_CA_AREAS)
    local = f"{rng.randint(0, 9)}{rng.choice(_CA_LETTERS)}{rng.randint(0, 9)}"
    return f"{rng.randint(10, 9999)} {rng.choice(_CA_STREETS)}, {city}, {province} {rng.choice(prefixes)} {local}"


#: (suburb, state, real postcode)
_AU_SUBURBS = (
    ("Harris Park", "NSW", "2150"), ("Parramatta", "NSW", "2150"), ("Blacktown", "NSW", "2148"),
    ("Point Cook", "VIC", "3030"), ("Tarneit", "VIC", "3029"), ("Glen Waverley", "VIC", "3150"),
)
_AU_STREETS = ("Wigram Street", "Station Street", "Church Street", "Victoria Road", "Park Road",
               "Sanctuary Lakes Boulevard")


def _australia(rng: random.Random) -> str:
    suburb, state, postcode = rng.choice(_AU_SUBURBS)
    unit = f"Unit {rng.randint(1, 20)}/" if rng.random() < 0.4 else ""
    return f"{unit}{rng.randint(1, 180)} {rng.choice(_AU_STREETS)}, {suburb} {state} {postcode}"


#: Qatar has no postcodes; addresses run by building, street and zone number. Zone numbers
#: are plausible, not checked against the official zone map.
_QATAR_AREAS = (("Al Mansoura", "25"), ("Najma", "26"), ("Al Sadd", "38"), ("Bin Mahmoud", "22"),
                ("Old Airport", "45"))


def _qatar(rng: random.Random) -> str:
    area, zone = rng.choice(_QATAR_AREAS)
    return f"Building {rng.randint(1, 120)}, Street {rng.randint(800, 990)}, Zone {zone}, {area}, Doha"


_FOREIGN_BUILDERS = {
    "United Arab Emirates": _uae,
    "United States": _us,
    "Saudi Arabia": _saudi,
    "United Kingdom": _uk,
    "Singapore": _singapore,
    "Canada": _canada,
    "Australia": _australia,
    "Qatar": _qatar,
}


def foreign_address(rng: random.Random, country: str) -> str:
    """A residential address in `country`, in that country's format, ending in the country name."""
    address = _FOREIGN_BUILDERS[country](rng)
    # Singapore's address already names the country, in front of its postcode.
    return address if country in address else f"{address}, {country}"


# --- Registered addresses for foreign companies --------------------------------------------
# Some are shared on purpose (free-zone business centres): a lead for Stage 5, not proof.

_BUSINESS_BUILDERS = {
    "United Arab Emirates": lambda rng: (
        f"Office {rng.randint(1, 38)}{rng.randint(0, 2)}{rng.randint(1, 9)}, "
        f"{rng.choice(('Bay Square Tower', 'Opal Tower', 'Silicon Oasis HQ', 'SAIF Zone Business Centre'))}, "
        f"{rng.choice(('Business Bay, Dubai', 'Jumeirah Lake Towers, Dubai', 'Dubai Silicon Oasis, Dubai', 'SAIF Zone, Sharjah'))}"
    ),
    "Hong Kong": lambda rng: (
        f"Unit {rng.randint(1, 20)}{rng.choice('ABCD')}, {rng.randint(3, 32)}/F, "
        f"{rng.choice(('Harbour Commercial Building', 'Kowloon Trade Centre', 'Wing On Plaza'))}, "
        f"{rng.choice(('Wan Chai', 'Tsim Sha Tsui', 'Kwun Tong', 'Sheung Wan'))}"
    ),
    # Ebene Cybercity is where Mauritius's management companies cluster. The 72201 postcode
    # is plausible for Ebene but was not individually verified.
    "Mauritius": lambda rng: (
        f"Level {rng.randint(2, 12)}, {rng.choice(('Tower A', 'Tower B', 'Cyber Tower 1', 'The Core'))}, "
        f"Ebene Cybercity, Ebene 72201"
    ),
    "Singapore": lambda rng: (
        f"{rng.randint(1, 180)} Robinson Road, #{rng.randint(3, 30):02d}-{rng.randint(1, 12):02d}, "
        f"Singapore 068{rng.randint(800, 999)}"
    ),
    "Cyprus": lambda rng: (
        f"{rng.randint(1, 120)} {rng.choice(('Spyrou Kyprianou Avenue', 'Arch. Makariou III Avenue'))}, "
        f"{rng.randint(3030, 3106)} Limassol"
    ),
    "Seychelles": lambda rng: (
        f"Suite {rng.randint(1, 40)}, {rng.choice(('Oliaji Trade Centre', 'Premier Building'))}, "
        f"Victoria, Mahe"
    ),
    "Malaysia": lambda rng: (
        f"Level {rng.randint(5, 30)}, {rng.choice(('Menara Kelana', 'Wisma Sentral'))}, "
        f"Jalan Sultan Ismail, 50250 Kuala Lumpur"
    ),
}


def foreign_business_address(rng: random.Random, country: str) -> str:
    """A company's registered address in `country` (residential format where none exists)."""
    builder = _BUSINESS_BUILDERS.get(country)
    if builder is None:
        return foreign_address(rng, country)
    return f"{builder(rng)}, {country}"
