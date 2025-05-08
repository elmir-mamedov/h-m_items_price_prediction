import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('mode.chained_assignment', None)  # Turns off the warning

df = pd.read_csv('handm.csv')
print(df.shape)
print("\n")

print(df.info())

print(df.head())

# Check the target value distribution

sns.histplot(x=df['price'])
#plt.ylim(0,)  # adjust 50 to whatever maximum makes sense for your data
plt.xlim(0, 150)  # adjust 50 to whatever maximum makes sense for your data
plt.show()

print("Null values:")
print(df.isnull().sum())

#checking unique values count
print("\nUnique:")
print(df.nunique())

df.drop(['colorShades'], axis=1, inplace=True)
df = df.dropna()

# Remove irrelevant features
df.drop(['Unnamed: 0'], axis=1, inplace=True) # Useless feature
df.drop(['productId'], axis=1, inplace=True) # Useless feature
df.drop(['brandName'], axis=1, inplace=True) # has only 1 value
df.drop(['url'], axis=1, inplace=True) # Useless feature
df.drop(['stockState'], axis=1, inplace=True) # has only 1 value
df.drop(['isOnline'], axis=1, inplace=True) # has only 1 value
df.drop(['comingSoon'], axis=1, inplace=True) # has only 1 value
#df.drop(['details'], axis=1, inplace=True) # could be helpful but needs to be vectorized
#df.drop(['materials'], axis=1, inplace=True) # for now
df.drop(['colors'], axis=1 , inplace=True) # Has better alternative (probably?)

# Binarize boolean
df["newArrival"] = df["newArrival"].astype(int)  # Convert boolean to 0/1

print(df.info())
print(df.nunique())
print(df.head(10))

# Break down colorName column

df['colorName'] = df['colorName'].str.lower() # set strings to lowercase

# Separate solid colors
colors = ['sand', 'lilac', 'plum', 'teal', 'taupe', 'burgundy', 'turquoise',
          'eggplant', 'taupe', 'white', 'black', 'green', 'blue', 'red', 'yellow',
          'orange', 'brown', 'grey', 'gray', 'purple', 'cream', 'beige',
          'khaki', 'pink', 'gold', 'ombre', 'navy', 'washed']


for index, row in df.iterrows():
    for color in colors:
        # Use regex to search for color within the colorName string
        if re.search(rf"{color}$", row['colorName']):
          df.loc[index, 'solid_colors'] = df.loc[index, 'colorName']
          df.loc[index, 'colorName'] = 'none'
          break

# Separate patterned clothes

patterns = ['glittery', 'glitter', 'stripes', 'ribbed', 'pinstriped', 'paisley',
            'patterned', 'striped', 'floral', 'plaid', 'checked', 'dotted',
            'melange', 'beads']


for index, row in df.iterrows():
    for pattern in patterns:
        # Use regex to search for color within the colorName string
        if re.search(rf"\b{pattern}\b", row['colorName']):
          df.loc[index, 'pattern'] = df.loc[index, 'colorName']
          df.loc[index, 'colorName'] = 'none'
          break

# Separate clothes with printing and delete the colorName column

for index, row in df.iterrows():
    if df.loc[index, 'colorName'] != 'none':
      df.loc[index, 'print'] = df.loc[index, 'colorName'] # no print/pattern

df.drop(['colorName'], axis=1, inplace=True)

# Break down mainCatCode column
binarize_cols = ['shoes', 'bottoms', 'jeans',
                 'tops', 'socks', 'outerwear', 'accessories', 'underwear', 'dresses',
                 'sportswear', 'loungewear', 'solid_colors', 'pattern', 'print']

df['mainCatCode'] = df['mainCatCode'].str.lower() # set strings to lowercase

for index, row in df.iterrows():
  if re.findall("^men", row['mainCatCode']):
    df.loc[index, 'gender'] = 'men'
  elif re.findall("women|ladies", row['mainCatCode']):
    df.loc[index, 'gender'] = 'ladies'
  else:
    df.loc[index, 'gender'] = 'neutral'



''' We can see that this third category is very small, but it can add
a percent or two to accuracy if we treat it properly. This
category seems to contist of items primarily cheaper than 12 bucks. We could
cut away the outliers and keep it.'''

df = pd.get_dummies(df, columns=["gender"], prefix="gender")  # One-hot encode gender


# Create premium column (if items is premium then 1, else 0)
premium_hist = []
count = 0
for index, row in df.iterrows():
  if re.findall("premium", row['mainCatCode']):
    df.loc[index, 'premium'] = 1
    premium_hist.append(df.loc[index, 'price'])
    count += 1

print(f"\nPremium count: {count}")


# Next big category is type type
# But we need some separation into item type categories


# shoes vary in price. E.g. heels must be way more expensive slippers, right?
# we can target encode this, it will not only give represent improtance,
# but also will reduce dimentionality compared to ohe.
df['slippers'] = 'none'
slippers_hist = []
slippers_list = ['slippers']
count = 0
for index, row in df.iterrows():
  for slipper in slippers_list:
    if re.search(rf"{slipper}", row['mainCatCode']):
      df.loc[index, 'slippers'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      slippers_hist.append(df.loc[index, 'price'])
      count += 1
      break
binarize_cols.append('slippers')
print(f"Slippers count: {count}")

df['shoes']='none'
shoes_hist = []
count = 0
for index, row in df.iterrows():
  if re.findall("shoes", row['mainCatCode']):
    df.loc[index, 'shoes'] = df.loc[index, 'mainCatCode']
    df.loc[index, 'mainCatCode'] = 'none'
    shoes_hist.append(df.loc[index, 'price'])
    count += 1
print(f"Shoes count (general): {count}")

count = 0
df['shoes_cat1'] = 'none'
shoes_cat1_list = ["sandals", "pumps", 'ballerinas']
for index, row in df.iterrows():
  for i in shoes_cat1_list:
    if re.search(rf"{i}", row['shoes']):
      df.loc[index, 'shoes_cat1'] = df.loc[index, 'shoes']
      df.loc[index, 'shoes'] = 'none'
      count += 1
      break
print(f"Shoes cat1 count: {count}")
binarize_cols.append("shoes_cat1")

count = 0
df['shoes_cat2'] = 'none'
shoes_cat2_list = ["dressed"]
for index, row in df.iterrows():
  for i in shoes_cat2_list:
    if re.search(rf"{i}", row['shoes']):
      df.loc[index, 'shoes_cat2'] = df.loc[index, 'shoes']
      df.loc[index, 'shoes'] = 'none'
      count += 1
      break
print(f"Shoes cat1 count: {count}")
binarize_cols.append("shoes_cat2")

count = 0
df['shoes_cat3'] = 'none'
shoes_cat3_list = ["selection"]
for index, row in df.iterrows():
  for i in shoes_cat3_list:
    if re.search(rf"{i}", row['shoes']):
      df.loc[index, 'shoes_cat3'] = df.loc[index, 'shoes']
      df.loc[index, 'shoes'] = 'none'
      count += 1
      break
print(f"Shoes cat1 count: {count}")
binarize_cols.append("shoes_cat3")

count = 0
df['boots'] = 'none'
boots_hist = []
boots_list = ["sneakers", "boots"]
for index, row in df.iterrows():
  for i in boots_list:
    if re.search(rf"{i}", row['shoes']):
      df.loc[index, 'boots'] = df.loc[index, 'shoes']
      df.loc[index, 'shoes'] = 'none'
      boots_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"boots count: {count}")
binarize_cols.append("boots")


# let's do the same with trousers
df['bottoms'] = 'none'
bottoms_hist = []
bottoms = ["trousers", "jeans", "shorts", "skirts", "leggings", "jumpsuits", "bottoms"]
count = 0
for index, row in df.iterrows():
  for bottom in bottoms:
    if re.search(rf"{bottom}", row['mainCatCode']):
      df.loc[index, 'bottoms'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      bottoms_hist.append(df.loc[index, 'price'])
      count += 1
      break
df['bottoms'] = df['bottoms'].fillna('').astype(str)
print(f"Bottoms count (general): {count}")

df['jeans'] = 'none'
jeans_hist = []
jeans = ['jeans']
count = 0
for index, row in df.iterrows():
  for j in jeans:
    if re.search(rf"{j}", row['bottoms']):
      df.loc[index, 'jeans'] = df.loc[index, 'bottoms']
      df.loc[index, 'bottoms'] = 'none'
      jeans_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Jeans count: {count}")

df['tops'] = 'none'
tops_hist = []
tops_list = [" Shirt", "T-shirt", "Tank Top",
             " Top", "umper", 'Sweater', 'Cardigan', "Bodysuit", 'Blouse']
count = 0
for index, row in df.iterrows():
  for top in tops_list:
    if re.search(rf"{top}", row['productName']):
      df.loc[index, 'tops'] = df.loc[index, 'productName']
      df.loc[index, 'mainCatCode'] = 'none'
      tops_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Tops count (general): {count}")

count = 0
df['ttop_tshirts'] = 'none'
ttop_tshirts_list = ["Tank Top", "T-shirt"]
for index, row in df.iterrows():
  for i in ttop_tshirts_list:
    if re.search(rf"{i}", row['tops']):
      df.loc[index, 'ttop_tshirts'] = df.loc[index, 'tops']
      df.loc[index, 'tops'] = 'none'
      count += 1
      break
print(f"tank top / t-shirts count: {count}")
binarize_cols.append("ttop_tshirts")

count = 0
df['shirts_jumpers'] = 'none'
shirts_jumpers = [" Shirt", "umper", 'Sweater', 'Cardigan', 'Blouse']
for index, row in df.iterrows():
  for i in shirts_jumpers:
    if re.search(rf"{i}", row['tops']):
      df.loc[index, 'shirts_jumpers'] = df.loc[index, 'tops']
      df.loc[index, 'tops'] = 'none'
      count += 1
      break
print(f"Shirt_jumpers count: {count}")
binarize_cols.append('shirts_jumpers')

df['socks']='none'
socks_hist = []
socks_list = ["socks"]
count = 0
for index, row in df.iterrows():
  for socks in socks_list:
    if re.search(rf"{socks}", row['mainCatCode']):
      df.loc[index, 'socks'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      socks_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Socks count: {count}")

df['outerwear'] = 'none'
outerwear_hist = []
outerwear_list = [ 'hoodies', 'blazer', 'jacket', 'bombers', 'parkas', 'coats', 'anoraks', 'outerwear']
count = 0
for index, row in df.iterrows():
  for outerwear in outerwear_list:
    if re.search(rf"{outerwear}", row['mainCatCode']) or \
           re.search(rf"{outerwear}", str(row['details']), re.IGNORECASE):
      df.loc[index, 'outerwear'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      outerwear_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Outerwear count (general): {count}")

blazers_hist = []
blazers_list = ['blazer']
count = 0
for index, row in df.iterrows():
  for blazer in blazers_list:
    if re.search(rf"{blazer}", row['outerwear']):
      df.loc[index, 'blazers'] = df.loc[index, 'outerwear']
      df.loc[index, 'outerwear'] = 'none'
      blazers_hist.append(df.loc[index, 'price'])
      count += 1
      break
binarize_cols.append('blazers')
print(f"Blazers count: {count}")


medium_outerwear_hist = []
medium_outerwear_list = [ 'hoodies', 'bombers']
count = 0
for index, row in df.iterrows():
  for outerwear in medium_outerwear_list:
    if re.search(rf"{outerwear}", row['outerwear']):
      df.loc[index, 'medium_outerwear'] = df.loc[index, 'outerwear']
      df.loc[index, 'outerwear'] = 'none'
      outerwear_hist.append(df.loc[index, 'price'])
      count += 1
      break
binarize_cols.append('medium_outerwear')
print(f"Medium outerwear count: {count}")

df['accessories'] = 'none'
accessories_hist = []
accessories = ["accessories", "hats", "bags", "sunglasses", "beauty", "care"]
count = 0
for index, row in df.iterrows():
  for accessory in accessories:
    if re.search(rf"{accessory}", row['mainCatCode']):
      df.loc[index, 'accessories'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      accessories_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Accessories count: {count}")

df['underwear']='none'
underwear_hist = []
underwear_list = ["lingerie", "knickers", "trunks",
                  "boxers", "bra", "basics", "underwear"
]
count = 0
for index, row in df.iterrows():
  for underwear in underwear_list:
    if re.search(rf"{underwear}", row['mainCatCode']):
      df.loc[index, 'underwear'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      underwear_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Underwear count: {count}")

df['dresses'] = 'none'
dresses_hist = []
dresses_list = ["dress"]
count = 0
for index, row in df.iterrows():
  for dress in dresses_list:
    if re.search(rf"{dress}", row['mainCatCode']):
      df.loc[index, 'dresses'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      dresses_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Dresses count: {count}")

df['sportswear'] = 'none'
sportswear_hist = []
sportswear_list = ["hiking", "skiing", "yoga", "watersports", "racketsports",
                   "training"
]
count = 0
for index, row in df.iterrows():
  for sportswear in sportswear_list:
    if re.search(rf"{sportswear}", row['mainCatCode']):
      df.loc[index, 'sportswear'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      sportswear_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Sportswear count: {count}")

df['loungewear'] = 'none'
loungewear_hist = []
loungewear_list = ["loungewear", "nightwear", "pyjamas", "nursing", "nightslips"]
count = 0
for index, row in df.iterrows():
  for loungewear in loungewear_list:
    if re.search(rf"{loungewear}", row['mainCatCode']):
      df.loc[index, 'loungewear'] = df.loc[index, 'mainCatCode']
      df.loc[index, 'mainCatCode'] = 'none'
      loungewear_hist.append(df.loc[index, 'price'])
      count += 1
      break
print(f"Loungewear count: {count}")


df['misc'] = 'none'
misc_hist = []
misc_list = ["ladies_divided", 'sweamwear']
count = 0
for index, row in df.iterrows():
  for i in misc_list:
    misc_hist.append(df.loc[index, 'price'])
    count += 1
    df.loc[index, 'misc'] = df.loc[index, 'mainCatCode']
    break

print(f"Misc count: {count}")
binarize_cols.append('misc')


# Since the biggest count is for ladies_divided and it ranges from 0 to 50
# I have no idea what to do with the rest of items. I dont want to create
# another features just for 100 items, but we'll see. If model doesn't
# perform well, we can try to do something with these features.
# But for now i will drop them.
print(df['mainCatCode'].value_counts())
df = df.drop(['mainCatCode'], axis=1)

plt.figure(figsize=(8, 5))
plt.xlim(0,300)
plt.hist(misc_hist, bins = 200)
plt.xlabel("Price")
plt.title("Price histogram")
plt.show()

items_category = 'sportswear'
# THIS DOESNT WORK (PRINT NONE ISNTEAD OF CAT NAME)
cat1, cat2, cat3 = [], [], []
categories = [cat1, cat2, cat3]
for cat, tresh in enumerate([(0,50), (50,150), (150,5000)], start=1):
    for index, row in df.iterrows():
        price = df.loc[index, 'price']
        items = df.loc[index, items_category] + '\n' + df.loc[index, 'details'] + \
            df.loc[index, 'materials']
        if tresh[0] < price < tresh[1] and items != 'none' :
            categories[cat - 1].append((items, price))

# sort each category by price
for i in range(len(categories)):
    categories[i] = sorted(categories[i], key=lambda item: item[1])

# print
for cat, tresh in enumerate([(0,50), (50,150), (150,5000)], start=1):
    print('*********************************')
    print(f'CATEGORIE {cat}:')
    for key, value in categories[cat - 1]:
        print(f"\n\n{key}. \nPRICE: {value}")
    print('\n\n*********************************')

"""# *** if i create a binary feature 'boots' from binary feature 'shoes', should i set boots to 1 and shoes to 0 , or should i keep both 1 to predict price better?***"""

plt.figure(figsize=(8, 5))
#plt.hist(boots_hist, bins = 50)
plt.hist(dresses_hist, bins = 150)
plt.xlabel("Price")
plt.legend(["dresses", ""])
plt.title(" Items price histogram")
plt.show()

plt.figure(figsize=(8, 5))
plt.xlim(0, 150)
#plt.hist(bottoms_hist, bins = 150)
plt.hist(shoes_hist, bins = 150)
plt.hist(slippers_hist, bins = 150)
#plt.hist(accessories_hist, bins = int(count/4))
plt.legend(["Shoes", "Slippers"])
plt.xlabel("Price")
plt.title("Shoes and slippers Price Histogram")
plt.show()

def extract_main_composition(text, threshold=15):
    # match lines like "Cotton 62%" or "Polyester 38%"
    matches = re.findall(r'([A-Za-z ]+?)\s+(\d+)%', text.split('ADDITIONAL MATERIAL INFORMATION')[0])
    return {material.strip(): int(percent) for material, percent in matches if int(percent) >= threshold}
    # text.split()[0] split the text into main info (which is before ADD.MAT.INFO)
    # and the rest. [0] returns the first part. i.e. main info
    # findall finds two match groups ([A-Za-z ]+?) and (\d+) i.e. things in brackets
    # findall puts everything into matches variable as a list of tuples
    # we then convert it into dictionary and return
    # material.strip() removes any leading or trailing whitespace

    # added threshold for composition %

df['main_materials'] = df['materials'].apply(lambda x: extract_main_composition(x, threshold=50))


'''for index, row in df.iterrows():
  print(f"\n{df.loc[index, 'main_materials']}")'''

print(type(df.loc[0, 'main_materials']))

'''materials = []
for index, row in df.iterrows():
  #print(f"\n{df.loc[index, 'main_materials']}")
  materials.extend(list((df.loc[index, 'main_materials']).keys()))
materials = set(materials)
for m in materials:
  print(m)'''

high_tier = [
    "Leather",
    "Suede",
    "Alpaca",
    "Cashmere",
    "Silk",
    "Mohair",
    "Wool",
    "Duck down grey",
    "Duck feather",
    "Reprocessed down",
    "Reprocessed feather",
    "Fresh water pearl",
    "Stainless steel",


    "Nappa",
    "Coated leather",
    "Regenerated leather"
]

mid_tier = [
    "Brass",
    "Aluminium",

    "Zinc alloy",
    "Linen",
    "Modal",
    "Lyocell",

    "Cotton",
    "Elastomultiester",





    "Polycarbonate", #(used in eyewear, some accessories — sturdier plastic)
    "Polymethyl methacrylate", # (aka acrylic glass — often for jewelry, mid-tier accessories)


    "Polyhydroxybutyrate", # (bioplastic — niche, sustainable, usually high-end or experimental)
    "Silicone",

    "Jute",
    "Birch",
    "Beech",
    "Glass",
    "Schima",
    "Wood"
]

low_tier = [
    "Ethylene Vinyl Acetate",
    "Polyamide",
    "Thermoplastic Polyurethane",
    "Acrylic",
    "Rayon",
    "Polyester",
    "Thermoplastic rubber",
    "Natural rubber",
    "PCTG",
    "PETG",
    "Polyethylene terephthalate",
    "Polybutylene terephthalate",
    "Resin",
    "Other fibres",
    "Modacrylic",
    "Plastic",
    "Polypropylene",
    "Polyurethane",
    "Polyethylene",
    "Polystyrene",
    "Polystyrene foam",
    "Styrene",
    "Metal",
    "Iron",
    "Steel",
    "Zinc",
    "Magnet",
    "Metallic fiber",
    "Spandex",
    "Paper",
    "Rubber",
    "Acetate",
    "Elastodiene",
]

material_tiers = [
    high_tier,
    mid_tier,
    low_tier
]

for index, row in df.iterrows():
    materials = row['main_materials']
    materials = " ".join(materials.keys())

    for tier_number, tier in enumerate(material_tiers, start=1):
        for m in tier:
            if re.search(rf"{m}", materials, flags=re.IGNORECASE):
                if tier_number == 1:
                    df.loc[index, 'high_tier'] = 1
                elif tier_number == 2:
                    df.loc[index, 'mid_tier'] = 1
                elif tier_number == 3:
                    df.loc[index, 'low_tier'] = 1
                break

# reshape the data to long format
melted = df.melt(
    id_vars='price',
    value_vars=['high_tier', 'mid_tier', 'low_tier'],
    var_name='tier',
    value_name='present'
)

# filter only rows where material is present (== 1)
melted = melted[melted['present'] == 1]

# plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='tier', y='price', data=melted)
plt.ylim(0, 400)
plt.xlabel('Material Tier')
plt.ylabel('Price')
plt.title('Price Distribution by Material Tier')
plt.show()

print(df.columns)

df = df.drop(['materials'], axis=1)
#df = df.drop(['details'], axis=1)
df = df.drop(['productName'], axis=1)
df = df.drop(['main_materials'], axis=1)



fillna_columns = ['high_tier', 'mid_tier', 'low_tier', 'premium']
for column in fillna_columns:
  df[column] = df[column].fillna(0)

# Binarizer

def binarize_column(val):
    if isinstance(val, str):
        return 0 if val.strip().lower() == 'none' else 1
    return 0

for column in binarize_cols:
    df[column] = df[column].apply(binarize_column)

# Log-transform the price
df['price'] = np.log1p(df['price'])  # log1p handles zero values safely

# Add TF-IDF for details column
tfidf = TfidfVectorizer(max_features=1000, stop_words='english') # stop_words='english' will ignore words like "in", "of" "the" "a" etc.
tfidf_matrix = tfidf.fit_transform(df['details'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
tfidf_df = tfidf_df.add_prefix('details_')
df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# we don't need scaler because all the values are in range [0 , 1]

df = df.drop(['details'], axis=1)

df.to_csv('hm_preprocessed.csv', index=False)
