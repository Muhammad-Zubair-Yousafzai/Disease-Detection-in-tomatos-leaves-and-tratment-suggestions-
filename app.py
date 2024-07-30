from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import h5py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Define classes and treatment suggestions
label = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
         'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
         'Not in our predict list']

fertilizer = {
    'Tomato___Bacterial_spot': ["What Cause it: Bacterial spot occurs worldwide and is one of the most devastating diseases on pepper and tomato crops grown in warm, moist environments. The pathogen can survive in association with seeds, either externally or internally as well as on specific weeds and later spreads through rain or overhead irrigation. It enters the plant through leaf pores and wounds. Optimum temperatures range from 25 to 30°C. Once the crop is infected, the disease is very difficult to control and can lead to total crop losses.","Organic Control: Bacterial Spot is very difficult and expensive to treat. If the disease occurs early in the season, consider destroying the entire crop. Copper-containing bactericides provide a protective cover on foliage and fruit. Bacterial viruses (bacteriophages) that specifically kill the bacteria are available. Submerge seeds for one minute in 1.3% sodium hypochlorite or in hot water (50°C) for 25 minutes.","Chemical Control: Always consider an integrated approach with preventive measures together with biological treatments if available. Copper-containing bactericides can be used as a protectant and give partial disease control. Application at the first sign of disease and then at 10- to 14-day intervals when warm, moist conditions prevail. The active ingredients copper and mancozeb give a better protection."],
             'Tomato___Early_blight': ["What Cause it: Symptoms are caused by Alternaria solani, a fungus that overwinters on infected crop debris in soil or on alternative hosts. Purchased seeds or seedlings may also be already contaminated. Lower leaves often get infected when in contact with contaminated soil. Warm temperatures (24- 29°C) and high humidity (90%) favor development of the disease. A long wet period (or alternating wet/dry weather) enhances the production of spores, which may be spread via wind, splashing rain or overhead irrigation. Tubers harvested green or in wet conditions are particularly susceptible to an infection. It often strikes after a period of heavy rainfall and is particularly destructive in tropical and subtropical areas.","Organic Control: Application of products based on Bacillus subtilis or copper-based fungicides registered as organic can treat this disease.","Chemical Control: Always consider an integrated approach with preventive measures and biological treatments if available. There are numerous fungicides on the market for controlling early blight. Fungicides based on or combinations of azoxystrobin, pyraclostrobin, difenoconazole, boscalid, chlorothalonil, fenamidone, maneb, mancozeb, trifloxystrobin, and ziram can be used. Rotation of different chemical compounds is recommended. Apply treatments in a timely manner, taking into account weather conditions. Check carefully the preharvest interval at which you can harvest safely after the application of these products."],
             'Tomato___Late_blight': ["What Cause it: The risk of infection is highest in midsummer. The fungus enters the plant via wounds and rips in the skin. Temperature and moisture are the most important environmental factors affecting the development of the disease. Late blight fungi grow best in high relative humidities (around 90%) and in temperature ranges of 18 to 26°C. Warm and dry summer weather can bring the spread of the disease to a halt.","Organic Control: At this point, there is no biological control of known efficacy against late blight. To avoid spreading, remove and destroy plants around the infected spot immediately and do not compost infected plant material.","Chemical Control: Always consider an integrated approach with preventive measures together with biological treatments if available. Use fungicide sprays based on mandipropamid, chlorothalonil, fluazinam, mancozeb to combat late blight. Fungicides are generally needed only if the disease appears during a time of year when rain is likely or overhead irrigation is practiced."],
             'Tomato___Leaf_Mold': ["What Cause it: The symptoms are caused by the fungus Mycovellosiella fulva, whose spores can survive without a host for 6 months to a year at room temperature (non-obligate). Prolonged leaf moisture and humidities above 85% favor the germination of spores. The temperature must be between 4° to 34 °C for spores to germinate, with an optimum temperature at 24- 26°C. Dry conditions and the absence of free water on leaves impair germination. The symptoms usually start to appear 10 days after inoculation with the development of spots on both sides of the leaf blade. On the underside, a large number of spore-producing structures are formed and these spores are easily spread from plant to plant by the wind and water splashing, but also on tools, clothing of workers and insects. The pathogen usually infects the leaves by penetrating through stomata in a high humidity level.","Organic Control: Seed treatment with hot water (25 minutes at 122 °F or 50 °C) is recommended to avoid the pathogen on seeds. The fungi Acremonium strictum, Dicyma pulvinata, Trichoderma harzianum or T. viride and Trichothecium roseum are antagonistic to M. fulva and could be used to reduce its spread. In greenhouse trials the growth of M. fulva on tomatoes was inhibited by A. strictum, Trichoderma viride strain 3 and T. roseum by 53, 66 and 84% respectively. In small arms, apple-cider, garlic or milk sprays and vinegar mix can be used to treat the mold.","Chemical Control: Always consider an integrated approach with preventive measures together with biological treatments if available. Applications should be made prior to infection when environmental conditions are optimal for the development of the disease. Recommended compounds in field use are chlorothalonil, maneb, mancozeb and copper formulations. For greenhouses, difenoconazole, mandipropamid, cymoxanil, famoxadone and cyprodinil are recommended."],
             'Tomato___Septoria_leaf_spot': ["What Cause it: The symptoms are caused by the fungus Mycosphaerella fragariae, which overwinter on infected leaf debris on the soil. During the spring, they resume growth and start to produce spores that are spread to the lower leaves of the neighboring plants. Spores that land onto the leaf lamina form germ tubes which penetrate through natural openings on upper and lower surfaces of leaves. As it grows, the fungus produces clusters of new spores that are carried to new leaves by rain splashes and wind. Human or machinery activity in the field can also be a source of contamination. The fruits are generally not directly affected, but the loss of foliage can reduce their quality and the yield. Disease development is favored by cool daytime temperatures (around 25 °C) and cold nighttime temperatures, high relative humidity, and prolonged leaf wetness. Bad cultural practices, such as short spacing between plants, can also increases the risk of infection.","Organic Control: Solutions containing the bacterium Bacillus cereus and the yeast Saccharomyces boulardii were used with the same efficiency as fungicides for disease control in laboratory tests. However, these products still need to be tested in large field experiments.","Chemical Control: Always consider an integrated approach with preventive measures together with biological treatments if available. This disease is hard to control because plants can start to experience the effect of the disease prior to showing any symptoms. Fungicides based on chlorothalonil, myclobutanil or triflumizole can be used to control common leaf spot disease on strawberries after appearance of first symptoms. Treatments should be done early in the spring or immediately after renovation and it is recommended to spray at intervals of about 2 weeks."],
             'Tomato___Spider_mites Two-spotted_spider_mite': ["What Cause it: Damage is caused by spider mites from the genus Tetranychus, mainly T. urticae and T. cinnabarinus. The adult female is 0.6 mm long, pale green with two darker patches on its oval body, and long hairs on the back. Overwintering females are reddish. In spring, the females lay globular and translucent eggs on the underside of the leaves. The nymphs are pale green with darker markings on the dorsal side. The mites protect themselves with a cocoon on the underside of the leaf blades. The spider mite thrives in dry and hot climates and will produce up to 7 generations in one year in these conditions. There is a wide range of alternative hosts, including weeds.","Organic Control: In case of minor infestation, simply wash off the mites and remove the affected leaves. Use preparations based on rapeseed, basil, soybean and neem oils to spray leaves thoroughly and reduce populations of T. urticae. Also try garlic tea, nettle slurry or insecticidal soap solutions to control the population. In fields, employ host-specific biological control with predatory mites (for example Phytoseiulus persimilis) or the biological pesticide Bacillus thuringiensis. A second spray treatment application 2 to 3 days after the initial treatment is necessary.","Chemical Control: Always consider an integrated approach with preventive measures together with biological treatments if available. Spider mites are very difficult to control with acaricides because most populations develop resistance to different chemicals after a few years of use. Choose chemical control agents carefully so that they do not disrupt the population of predators. Fungicides based on wettable sulfur (3 g/l), spiromesifen (1 ml/l), dicofol (5 ml/l) or abamectin can be used for example (dilution in water). A second spray treatment application 2 to 3 days after the initial treatment is necessary."],
             'Tomato___Target_Spot': ["What Cause it: Tomato spotted wilt virus (TSWV) is transmitted by various species of thrips, including the western flower thrips (Frankliniella occidentalis), the onion thrips (Thrips tabaci) and the chili thrips (Scirtothrips dorsalis). TSWV is also active in the thrips vector and can transmit it persistently. Nymphs that acquire the virus by feeding on infected plants will retain the ability to transmit it for the remainder of their lives. However, TSWV cannot be passed from infected females to the eggs. The virus has a very wide host range, including tomato, pepper, potato, tobacco, lettuce, and many other plants.","Organic Control: Some predatory mites feed on larvae or pupae of thrips and are commercially available. For varieties that attack the leaves and not the flowers, try neem oil or spinosad, especially on the undersides of the leaves. Spinosad application is very effective but can be toxic to certain natural enemies (e.g., predatory mites, syrphid fly larvae, bees) and should be avoided during flowering time. In case of flower thrips infestation, some predatory mites or green lacewing larvae could be used. A combination of garlic extracts with some insecticides also seem to work well.","Chemical Control: Always consider an integrated approach of preventive measures together with biological treatments if available. Due to the high reproductive rates and their life cycles, thrips have developed resistance to different classes of pesticides. Effective contact insecticides include azadirachtin or pyrethroids, which in many products are combined with piperonyl butoxide to enhance their effect."],
             'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ["What Cause it: TYLCV is not seed-borne and is not transmitted mechanically. It is spread by whiteflies of the Bemisia tabaci species. These whiteflies feed on the lower leaf surface of a number of plants and are attracted by young tender plants. The whole infection cycle can take place in about 24 hours and is favored by dry weather with high temperatures.","Organic Control: There is no treatment against TYLCV. Control the whitefly population to avoid the infection with the virus.","Chemical Control: Once infected with the virus, there are no treatments against the infection. Control the whitefly population to avoid the infection with the virus. Insecticides of the family of the pyrethroids used as soil drenches or spray during the seedling stage can reduce the population of whiteflies. However, their extensive use might promote resistance development in whitefly populations."],
             'Tomato___Tomato_mosaic_virus': ["What Cause it: The virus is transmitted in a persistent manner by the whitefly Bemisia tabaci. Plants can also be infected via mechanical injury during field work. The virus is not passed from plant to plant in a systematic manner, nor is it seed- or pollen-borne. tomato usually get infected when volunteer plants or host weeds are present in the field. The virus multiplies in the transport tissues of plants, which explains why veins are affected first. The appearance of visible symptoms and their severity is favored by elevated temperatures around 28°C. Cooler conditions (around 22°C) may delay the multiplication of the virus and the development of symptoms.","Organic Control: Application of leaf extracts of Iresine herbstii (Herbst's bloodleaf) and Phytolacca thyrsiflora can partly inhibit virus infection and results in less incidence in the field. Extracts of the beneficial fungus Beauveria bassiana have insecticide properties against the adults, eggs and nymphs of bemisia tabaci.","Chemical Control: Always consider an integrated approach with preventive measures together with biological treatments available. Chemical control of viral infections is not possible. As for the control of whiteflies, very few treatments are effective."],
             'Tomato___healthy': ["No Disease Detected","No Teartment Required ","No Chemical Required"],
             'Not in our predict list': "Not in our predict list, Please check the predict list before upload" 
}

# Load your saved PyTorch model in HDF5 format
model_path = 'model.h5'

def load_h5_model(model_path):
    with h5py.File(model_path, 'r') as f:
        # Read the model's state dictionary from HDF5
        state_dict = {}
        for key in f.keys():
            if f[key].shape == ():  # Check if the dataset is scalar
                state_dict[key] = torch.tensor(f[key][()])
            else:
                state_dict[key] = torch.tensor(f[key][:])
    
    # Instantiate your model and load state_dict
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Change the number of outputs to 10
    model.load_state_dict(state_dict)
    return model

# Load the model
model = load_h5_model(model_path)
model.eval()

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Function to predict and get treatment suggestion
def predict_image(image_path):
    image_tensor = preprocess_image(image_path)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    class_index = predicted.item()
    class_name = label[class_index]
    
    # Get treatment suggestions
    if class_name in fertilizer:
        suggestions = fertilizer[class_name]
    else:
        suggestions = fertilizer['Not in our predict list']
    
    return class_name, suggestions

# Route to handle index page and file upload
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            import os
            upload_folder = os.path.join(os.path.dirname(os.path.abspath('')), 'static', 'uploads')

            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            class_name, suggestions = predict_image(file_path)
            return render_template('index.html', image=file_path, result=class_name, cause=suggestions[0], org=suggestions[1], chem=suggestions[2])

if __name__ == '__main__':
    app.run(debug=False)
