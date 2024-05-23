# coding=utf-8
import random

import numpy as np
# from transformers import AutoTokenizer, AutoModel
# import torch

#
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#
#
# # Sentences we want sentence embeddings for
# # sentences = ['This is an example sentence', 'Each sentence is converted']
# sentences = ["""
#          Supervised learning is the machine learning task of learning a function that
#          maps an input to an output based on example input-output pairs. It infers a
#          function from labeled training data consisting of a set of training examples.
#          In supervised learning, each example is a pair consisting of an input object
#          (typically a vector) and a desired output value (also called the supervisory signal).
#          A supervised learning algorithm analyzes the training data and produces an inferred function,
#          which can be used for mapping new examples. An optimal scenario will allow for the
#          algorithm to correctly determine the class labels for unseen instances. This requires
#          the learning algorithm to generalize from the training data to unseen situations in a
#          'reasonable' way (see inductive bias).
#       """,
#              """
#              My Corvette Moment? When I Took the Mighty ZR1 All the Way to 200 MPH
#              It's fitting graces the cover of the 70th anniversary print issue of magazine ( ). We've grown up together, you see. "We wanted a magazine that would interest the foreign car exponent, the sports car enthusiast, the custom car fan, and also be equally interesting to the stock car owner," original editor-in-chief Walt Woron wrote as he put the finishing touches on the September 1949 issue. "A magazine that brings you the trends of the automotive field: designs of the future, what's new in motoring, news from the Continent, trends in design." founder Robert Petersen's personal connection with Southern California race car builder Frank Kurtis perhaps explains why he chose as the first cover car for his new magazine—rather than, say, a sedan, America's top-selling car that year. But the choice was also an eerily prescient confirmation of 's mission statement. Within two years of the Kurtis appearing on our cover, a senior GM executive in Detroit had instigated a secret backroom program code-named Project Opel, a proposal for a fiberglass-bodied sports car that, like the Kurtis, used many regular production car components under its shapely skin. The GM exec's name? Harley Earl. And the car? Well, it first came to the public's attention as the EX-122, one of the stars of GM's 1953 Motorama Show at New York's Waldorf Astoria Hotel. But you know it better as the original . Frank Kurtis had the idea. GM had the money. Today is more than just a magazine. It's , linear TV channels, a website, and a social media phenomenon—an automotive content creator and curator with an audience that now spans the globe. has grown up. So, too, has the Chevrolet Corvette. The C8 is still America's Own Sports Car, but with its mid-engine layout, it's built to take on all comers, from Italy's to Britain's and Germany's .  """]
#
# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('D:\\AI\\model\\distilbert-base-uncased')
# model = AutoModel.from_pretrained('D:\\AI\\model\\paraphrase-MiniLM-L6-v2')
#
# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
#
# # Perform pooling. In this case, max pooling.
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#
# print("Sentence embeddings:")
# print(sentence_embeddings)


#
from keybert import KeyBERT
from utils.dataset_utils import clean_df
#
#
# # def g_new_word():
# doc = [ """Trying to Make a Ram 3500 as Quick as a Viper Requires Some Disassembly
#       Some blasphemy was involved in going to work on an $87,00 2019 Ram 3500 with wrenches and pliers. Track workers looked on with amusement and concern as we pulled the truck down to its skeletal self. Piles of brand-new, metallic-red sheetmetal and deep-black leather were strewn out under a cloudy central California sky a few feet from Auto Club Famoso Raceway’s quarter-mile. If you’re wondering what fueled the explosion of coal-rolling, lifted diesel pickups over the past few years, take a good look at this truck. Ram will now sell you a pickup with 400 hp and 1000 lb-ft of torque in factory trim. One thousand. That number can only be bested by the likes of the Bugatti Chiron, and it is part of the recipe that lets the truck tow up to 31,500 pounds. That’s seven Chirons and their well-moneyed occupants. It’s also far in excess of what you can legally lug behind a truck without a commercial driver’s license in most states. That’s 26,000 pounds, truck and trailer combined weight. The really crazy part? The diesel aftermarket has been hitting those power numbers with simple ECU reflashes for decades. The only difference is that the 2019 Ram does it with a factory warranty and without spewing clouds of black smoke. It’s not Lamborghinis that fill the heads of our nation’s farmland youth. It’s the new Cummins. This machine may be the poster child for capability creep, a perfect example of our age’s obsession with progress. The Ram is also a stunning piece of engineering, a tow rig that’s just as fascinating as anything you’d throw on the trailer behind it. Looking at its spec sheet got us thinking about another moonshot Chrysler: the first generation Viper. Its 8.0-liter V-10 managed 400 hp and 450 lb-ft of torque when it debuted in 1991, figures that were good enough to rip the car down the quarter-mile in 12.9 seconds. The Ram’s 6.7-liter I-6 turbodiesel Cummins engine puts the Viper’s power numbers in the dirt, but what about that last bit? Could we get this massive, one-ton pickup down the drag strip as quickly as a Viper without doing anything under the hood? This was a dumb question made dumber by the truck at hand: a four-wheel-drive, four-door, long-bed dualie loaded with every option, including five onboard cameras, a massive tombstone infotainment screen, and a factory fifth-wheel hitch. At more than 8500 pounds, the big Ram carries more mass than two Vipers. But, hey, you hit with the fists you’ve got, not the ones you wish you had. If any engine is up to the task, it’s this one. Cummins is a company synonymous with immortal engines, workhorses unfazed by neglect or abuse, capable of mind-boggling power and longevity. It’s not uncommon to see an old 5.9 still rattling away with more than 400,000 miles on the odometer, a thumb in the eye of anyone who’s ever cast a sidelong glance at a Chrysler. Where most truck owners afford their chosen brand the kind of loyalty reserved for mothers and other deities, a Cummins gets respect from everyone, regardless of the keys in their pocket. “Hell of a motor” is the usual refrain. The take rate is insane. Of all heavy-duty Rams that roll off the showroom floor, 80 percent leave with a Cummins badge on the fender. Years ago, when the company unveiled its 850-lb-ft max-tow powerplant, I cornered an engineer. I wanted to know if they were getting close to some sort of power ceiling, if the engine’s architecture could take much more. The guy looked at me like I’d asked him if the sun was made of cheese. “No,” he laughed. “When they build a transmission that can stand up to what we can produce, we’ll have the power for them.” That was the old 6.7-liter straight-six. This engine is something new. Cummins abandoned the cast-iron block in favor of compacted graphite iron, which is lighter and 75 percent stronger than the old material. That pulled 60 pounds from the engine and allows the big six to withstand higher cylinder pressures. Good thing, as there’s a new variable-geometry Holset/Cummins turbo bolted to the manifold that shoves 33 psi down the engine’s throat. There’s a new crank and forged rods to help deal with the extra boost, and the redesigned cast-iron cylinder head has hydraulic lifters for the first time. Ram’s been on a long quest to civilize its big diesel, and now the engine is quiet from behind the wheel, the familiar valvetrain clatter nearly banished. I miss it. That racket was the background to a wild and brilliant year for my family, when we turned a 2003 Dodge Ram 2500 with more than 280,000 miles on it into a camper and lived out of it. If I seem like a Cummins cheerleader, it is because I have seen what the engines are capable of. Over 11 months, we put more than 30,000 miles on that truck, every one of them overloaded, plenty off-road. The engine never blinked. Not scrambling through Big Bend National Park’s Black Gap Road nor pulling itself and a buddy’s derelict Tacoma up Eisenhower Pass at 11,000 feet, the Ram’s boost gauge pegged at 21 psi for 30 minutes straight. Hell of a motor. At Auto Club Famoso, we needed to set a baseline. The track was clean and prepped. The air was cool. I rolled the truck into the burnout box and set the rear wheels spinning. The tach started swinging, accompanied by the vague sensation that something dramatic was unfolding very far away. It was like riding out an earthquake from the next state over. Once in the staging beam, I ramped up the boost and let fly. I don’t know what I expected. All 1000 lb-ft of torque to take a seat on my chest, probably. Instead, it felt slow. First gear was a dog. Chrysler has finally put the manual transmission out to pasture, and your only option for the max-tow package is now an Aisin six-speed automatic. Thrust arrived after second gear, the unladen truck bouncing down the track on its way to a devastating 16.723 seconds at 85.64 mph. Woof. For the first time, it was clear exactly how much power it takes to get this thing moving, to overcome all that weight. We had some work to do. Drag-racing lore says that for every 100 pounds you delete from a vehicle, you gain a tenth of a second in the quarter. By that math, in order to get the Ram down to the Viper’s 12.9-second time, we would have to pull 3820 pounds from the truck. It wasn’t just a tall order, it was outright impossible, but we busted out the tools anyhow, confident in at least one other trick we’d brought as a nuclear option: a set of Hoosier drag radials. First went that fifth-wheel hitch, and when we weighed it on our super accurate Walmart bathroom scales, it came out to 160 pounds by itself. Then went the tonneau cover. The tailgate followed, another 60 pounds. The rear bumper? Thirty more. Then the rear hitch. The spare tire alone was nearly 70 pounds. This was looking good. The thing is, once you start taking parts off of a machine, it’s hard to stop. And for all the changes the brand has brought to the Ram Heavy Duty in the 16 years between my truck and this one, there’s so much that’s the same. The bones beneath that drastically different skin are remarkably similar. I knew how to take the bed off. There are a million wires that run back there, way more than what’s on the older trucks. I know this because they make a distinct sinew-snapping sound when they’re ripped free by five guys in a parking lot. This truck’s owner winced at the sound but was committed to our foolishness. Then went the automatic running boards and doors. Every time we disconnected a wiring harness, the dash dinged in protest. “What are you doing, Dave?” It took this long for me to consider the possibility that the truck might refuse to start after finding itself skinned and gutted. New vehicles are uncomfortably self-aware, but we’d come too far. Out went the seats. It’s hard to say exactly how much weight we removed. The doors were good for around 360 pounds. I’d say we pulled the better part of half a Miata out of the truck. There was a short-circuit in my brain as I approached the opening that was formerly the driver’s door. I found myself reaching for a handle that was lying on the asphalt 10 feet away. Ingress had been significantly improved. I pressed the brake and thumbed the start button, and to my sheer joy, the Cummins fired. Bring on the apocalypse. I am Lord Humongous and this is my War Rig. Bow before my death tractor, ye of Viper Clan. All those precious noises I missed came bouncing off the pavement and into the cabin. That clatter like an old friend, the whoosh of the mammoth turbo. It’d been hours since our first run. The track surface was now stone-cold, but I could still hear the tires sticking to the layers of spent rubber as I rolled into the burnout box once again, the stale-wine smell of the track prep deep in my lungs as the back tires got to spinning. You want to feel fast? Fire a doorless dualie down a drag strip. There was burning rubber in my nose hairs; the wind grabbed at the seatbelts and flung the buckles out of the open cabin, the tires making a ray-gun skeeeeeeeeeeewwww racket as I tripped the beam at more than 90 mph. I was heartbroken as I rolled back to the staging lanes, the lights glowing with a 15.855 in my rearview. We tried a few more runs. First deleting the outer dualie wheels, rationalizing that what we lost in traction we’d make up for in wheel speed. We went slower. Then came my ace: those gorgeous Hoosiers in the tallest size they make. I’d scrounged up a set of cheap steel wheels hoping Ram hadn’t changed the lug pattern on the new axle. Hope in one hand, etc. I nearly cried when the wheels did not fit, our tires lying unsmoked in the pile of parts scattered across the staging lanes. We made one last pass, this time with the spare back in place, praying that the extra weight would lend some needed traction. It didn’t. That was when the first fat raindrops fell on the windshield. A sea of disassembled Ram stared back at me through the glass. Just like that, we were done. With water on the track, our day was over. In the quarter-mile, each second is a lifetime, and there were still three of them between our best run and the first-gen Viper’s published time. I know that in motorsports the steps to each victory are paved with a hundred thousand misses, an encyclopedia of lessons taught by life’s most stern and valuable teacher. This was just one of them. There’s no telling what we could have accomplished with a better driver and those sticky Hoosiers, let alone a lighter truck, but the fact that this mammoth machine can punch its way to a 15-second quarter is a minor miracle, another line in the Cummins legend. It would be hours before we had the truck reassembled and fully functional, thin cold mist falling in sheets into the sparse light cast by one generous street lamp. And while my hands were busy working fasteners and connectors, my mind kept running over the infinite variables that go into something as simple as moving quickly in a straight line for a mere quarter mile. We’d asked this Cummins to do something impossible, something so far beyond the scope of its intended use that in most cases, it wouldn’t be worth trying at all. But this truck and its engine have built an empire on the bones of what shouldn’t be possible. If it can pull 31,500 pounds, slog its way to half a million miles, and tear at the ground with 1000 lb-ft of torque, why shouldn’t it be able to chase down a first-generation Viper? Weight be damned. I’m not convinced we failed so much as made progress toward that 12.9-second quarter, and if another Ram 3500 shows up in my driveway, I’ll put those Hoosiers to use and make a little more."""
#       ]
           # """
           #      Supervised learning is the machine learning task of learning a function that
           #      maps an input to an output based on example input-output pairs. It infers a
           #      function from labeled training data consisting of a set of training examples.
           #      In supervised learning, each example is a pair consisting of an input object
           #      (typically a vector) and a desired output value (also called the supervisory signal).
           #      A supervised learning algorithm analyzes the training data and produces an inferred function,
           #      which can be used for mapping new examples. An optimal scenario will allow for the
           #      algorithm to correctly determine the class labels for unseen instances. This requires
           #      the learning algorithm to generalize from the training data to unseen situations in a
           #      'reasonable' way (see inductive bias).
           #   """,
           #    """Trying to Make a Ram 3500 as Quick as a Viper Requires Some Disassembly
           #    Some blasphemy was involved in going to work on an $87,00 2019 Ram 3500 with wrenches and pliers. Track workers looked on with amusement and concern as we pulled the truck down to its skeletal self. Piles of brand-new, metallic-red sheetmetal and deep-black leather were strewn out under a cloudy central California sky a few feet from Auto Club Famoso Raceway’s quarter-mile. If you’re wondering what fueled the explosion of coal-rolling, lifted diesel pickups over the past few years, take a good look at this truck. Ram will now sell you a pickup with 400 hp and 1000 lb-ft of torque in factory trim. One thousand. That number can only be bested by the likes of the Bugatti Chiron, and it is part of the recipe that lets the truck tow up to 31,500 pounds. That’s seven Chirons and their well-moneyed occupants. It’s also far in excess of what you can legally lug behind a truck without a commercial driver’s license in most states. That’s 26,000 pounds, truck and trailer combined weight. The really crazy part? The diesel aftermarket has been hitting those power numbers with simple ECU reflashes for decades. The only difference is that the 2019 Ram does it with a factory warranty and without spewing clouds of black smoke. It’s not Lamborghinis that fill the heads of our nation’s farmland youth. It’s the new Cummins. This machine may be the poster child for capability creep, a perfect example of our age’s obsession with progress. The Ram is also a stunning piece of engineering, a tow rig that’s just as fascinating as anything you’d throw on the trailer behind it. Looking at its spec sheet got us thinking about another moonshot Chrysler: the first generation Viper. Its 8.0-liter V-10 managed 400 hp and 450 lb-ft of torque when it debuted in 1991, figures that were good enough to rip the car down the quarter-mile in 12.9 seconds. The Ram’s 6.7-liter I-6 turbodiesel Cummins engine puts the Viper’s power numbers in the dirt, but what about that last bit? Could we get this massive, one-ton pickup down the drag strip as quickly as a Viper without doing anything under the hood? This was a dumb question made dumber by the truck at hand: a four-wheel-drive, four-door, long-bed dualie loaded with every option, including five onboard cameras, a massive tombstone infotainment screen, and a factory fifth-wheel hitch. At more than 8500 pounds, the big Ram carries more mass than two Vipers. But, hey, you hit with the fists you’ve got, not the ones you wish you had. If any engine is up to the task, it’s this one. Cummins is a company synonymous with immortal engines, workhorses unfazed by neglect or abuse, capable of mind-boggling power and longevity. It’s not uncommon to see an old 5.9 still rattling away with more than 400,000 miles on the odometer, a thumb in the eye of anyone who’s ever cast a sidelong glance at a Chrysler. Where most truck owners afford their chosen brand the kind of loyalty reserved for mothers and other deities, a Cummins gets respect from everyone, regardless of the keys in their pocket. “Hell of a motor” is the usual refrain. The take rate is insane. Of all heavy-duty Rams that roll off the showroom floor, 80 percent leave with a Cummins badge on the fender. Years ago, when the company unveiled its 850-lb-ft max-tow powerplant, I cornered an engineer. I wanted to know if they were getting close to some sort of power ceiling, if the engine’s architecture could take much more. The guy looked at me like I’d asked him if the sun was made of cheese. “No,” he laughed. “When they build a transmission that can stand up to what we can produce, we’ll have the power for them.” That was the old 6.7-liter straight-six. This engine is something new. Cummins abandoned the cast-iron block in favor of compacted graphite iron, which is lighter and 75 percent stronger than the old material. That pulled 60 pounds from the engine and allows the big six to withstand higher cylinder pressures. Good thing, as there’s a new variable-geometry Holset/Cummins turbo bolted to the manifold that shoves 33 psi down the engine’s throat. There’s a new crank and forged rods to help deal with the extra boost, and the redesigned cast-iron cylinder head has hydraulic lifters for the first time. Ram’s been on a long quest to civilize its big diesel, and now the engine is quiet from behind the wheel, the familiar valvetrain clatter nearly banished. I miss it. That racket was the background to a wild and brilliant year for my family, when we turned a 2003 Dodge Ram 2500 with more than 280,000 miles on it into a camper and lived out of it. If I seem like a Cummins cheerleader, it is because I have seen what the engines are capable of. Over 11 months, we put more than 30,000 miles on that truck, every one of them overloaded, plenty off-road. The engine never blinked. Not scrambling through Big Bend National Park’s Black Gap Road nor pulling itself and a buddy’s derelict Tacoma up Eisenhower Pass at 11,000 feet, the Ram’s boost gauge pegged at 21 psi for 30 minutes straight. Hell of a motor. At Auto Club Famoso, we needed to set a baseline. The track was clean and prepped. The air was cool. I rolled the truck into the burnout box and set the rear wheels spinning. The tach started swinging, accompanied by the vague sensation that something dramatic was unfolding very far away. It was like riding out an earthquake from the next state over. Once in the staging beam, I ramped up the boost and let fly. I don’t know what I expected. All 1000 lb-ft of torque to take a seat on my chest, probably. Instead, it felt slow. First gear was a dog. Chrysler has finally put the manual transmission out to pasture, and your only option for the max-tow package is now an Aisin six-speed automatic. Thrust arrived after second gear, the unladen truck bouncing down the track on its way to a devastating 16.723 seconds at 85.64 mph. Woof. For the first time, it was clear exactly how much power it takes to get this thing moving, to overcome all that weight. We had some work to do. Drag-racing lore says that for every 100 pounds you delete from a vehicle, you gain a tenth of a second in the quarter. By that math, in order to get the Ram down to the Viper’s 12.9-second time, we would have to pull 3820 pounds from the truck. It wasn’t just a tall order, it was outright impossible, but we busted out the tools anyhow, confident in at least one other trick we’d brought as a nuclear option: a set of Hoosier drag radials. First went that fifth-wheel hitch, and when we weighed it on our super accurate Walmart bathroom scales, it came out to 160 pounds by itself. Then went the tonneau cover. The tailgate followed, another 60 pounds. The rear bumper? Thirty more. Then the rear hitch. The spare tire alone was nearly 70 pounds. This was looking good. The thing is, once you start taking parts off of a machine, it’s hard to stop. And for all the changes the brand has brought to the Ram Heavy Duty in the 16 years between my truck and this one, there’s so much that’s the same. The bones beneath that drastically different skin are remarkably similar. I knew how to take the bed off. There are a million wires that run back there, way more than what’s on the older trucks. I know this because they make a distinct sinew-snapping sound when they’re ripped free by five guys in a parking lot. This truck’s owner winced at the sound but was committed to our foolishness. Then went the automatic running boards and doors. Every time we disconnected a wiring harness, the dash dinged in protest. “What are you doing, Dave?” It took this long for me to consider the possibility that the truck might refuse to start after finding itself skinned and gutted. New vehicles are uncomfortably self-aware, but we’d come too far. Out went the seats. It’s hard to say exactly how much weight we removed. The doors were good for around 360 pounds. I’d say we pulled the better part of half a Miata out of the truck. There was a short-circuit in my brain as I approached the opening that was formerly the driver’s door. I found myself reaching for a handle that was lying on the asphalt 10 feet away. Ingress had been significantly improved. I pressed the brake and thumbed the start button, and to my sheer joy, the Cummins fired. Bring on the apocalypse. I am Lord Humongous and this is my War Rig. Bow before my death tractor, ye of Viper Clan. All those precious noises I missed came bouncing off the pavement and into the cabin. That clatter like an old friend, the whoosh of the mammoth turbo. It’d been hours since our first run. The track surface was now stone-cold, but I could still hear the tires sticking to the layers of spent rubber as I rolled into the burnout box once again, the stale-wine smell of the track prep deep in my lungs as the back tires got to spinning. You want to feel fast? Fire a doorless dualie down a drag strip. There was burning rubber in my nose hairs; the wind grabbed at the seatbelts and flung the buckles out of the open cabin, the tires making a ray-gun skeeeeeeeeeeewwww racket as I tripped the beam at more than 90 mph. I was heartbroken as I rolled back to the staging lanes, the lights glowing with a 15.855 in my rearview. We tried a few more runs. First deleting the outer dualie wheels, rationalizing that what we lost in traction we’d make up for in wheel speed. We went slower. Then came my ace: those gorgeous Hoosiers in the tallest size they make. I’d scrounged up a set of cheap steel wheels hoping Ram hadn’t changed the lug pattern on the new axle. Hope in one hand, etc. I nearly cried when the wheels did not fit, our tires lying unsmoked in the pile of parts scattered across the staging lanes. We made one last pass, this time with the spare back in place, praying that the extra weight would lend some needed traction. It didn’t. That was when the first fat raindrops fell on the windshield. A sea of disassembled Ram stared back at me through the glass. Just like that, we were done. With water on the track, our day was over. In the quarter-mile, each second is a lifetime, and there were still three of them between our best run and the first-gen Viper’s published time. I know that in motorsports the steps to each victory are paved with a hundred thousand misses, an encyclopedia of lessons taught by life’s most stern and valuable teacher. This was just one of them. There’s no telling what we could have accomplished with a better driver and those sticky Hoosiers, let alone a lighter truck, but the fact that this mammoth machine can punch its way to a 15-second quarter is a minor miracle, another line in the Cummins legend. It would be hours before we had the truck reassembled and fully functional, thin cold mist falling in sheets into the sparse light cast by one generous street lamp. And while my hands were busy working fasteners and connectors, my mind kept running over the infinite variables that go into something as simple as moving quickly in a straight line for a mere quarter mile. We’d asked this Cummins to do something impossible, something so far beyond the scope of its intended use that in most cases, it wouldn’t be worth trying at all. But this truck and its engine have built an empire on the bones of what shouldn’t be possible. If it can pull 31,500 pounds, slog its way to half a million miles, and tear at the ground with 1000 lb-ft of torque, why shouldn’t it be able to chase down a first-generation Viper? Weight be damned. I’m not convinced we failed so much as made progress toward that 12.9-second quarter, and if another Ram 3500 shows up in my driveway, I’ll put those Hoosiers to use and make a little more."""
           #    ]
# kw_model = KeyBERT(model='D:\\AI\\model\\roberta-base')
#        keywords = kw_model.extract_keywords(doc)
#        print("keywords...")
#
#        print(concat_words(keywords)
# keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words='english')
# print(keywords)
#        print(concat_words(keywords))
#        keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english', use_maxsum=True,
#                                             nr_candidates=20, top_n=5)
#        print(concat_words(keywords))
#
#
# def concat_words(keyword):
#     words = []
#     for word in keyword:
#         w = [x[0] for x in word]
#         new_w = " ".join(w)
#         words.append(new_w)
#     return words
#
#
# if __name__=="__main__":
#        g_new_word()


# doc_embeddings, word_embeddings = kw_model.extract_embeddings(doc)
# print("doc_embeddings...")
# print(doc_embeddings.shape)

# import torch
# import pandas as pd
# from keybert import KeyBERT
# from utils.dataset_utils import clean_df
#
#
# def get_read(name="MIND15"):
#     df = clean_df(pd.read_csv("D:\\AI\\Graduation_Project\\model\\BATM\\dataset\\data\\" + name + ".csv", encoding="utf-8"))
#     # data = [df.iloc[:, 2], df.iloc[:, 9], df.iloc[:, 10]]
#     # print(len(data))
#     # print(data)
#     df["data"] = df.title + "\n" + df.body
#     # labels = df["category"].values.tolist()
#     # label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
#     # labels = [label_dict[x] for x in labels]
#     data = {}
#     data["text"] = df["data"]
#     data["label"] = df["category"].values.tolist()
#     data["split"] = df.iloc[:, 10]
#     return data
#
#
# def generate(data_loader,name="test_demo", model="distilbert-base-uncased", batch_size=5120):
#     model = KeyBERT(model='D:\\AI\\model\\' + model)
#     result = [["text", "label", "split"]]
#     labels = data_loader["label"]
#     split = data_loader["split"]
#     text = data_loader["text"]
#     length = len(text)
#     assert len(split)==len(labels)==len(text)
#     count = 1
#     for i in range(0, length, batch_size):
#         end = min(length, i + batch_size)
#         data = list(text[i:end])
#         keywords = model.extract_keywords(data, keyphrase_ngram_range=(1, 2), stop_words="english")
#         keywords = concat_words(keywords)
#         temp = [[keywords[j], labels[i + j], split[i + j]] for j in range(len(keywords))]
#         result.extend(temp)
#         if (end >= count * 10000):
#             print(end)
#             count += 1
#     # print(result)
#     df = pd.DataFrame(result)
#     csv_dile_path = 'D:\\AI\\Graduation_Project\\model\\BATM\\dataset\\data\\' + name + '.csv'
#     df.to_csv(csv_dile_path, index=False, header=False)
#     print("CSV 文件已保存到:", csv_dile_path)
#
#
# def concat_words(keyword):
#     words = []
#     for word in keyword:
#         w = [x[0] for x in word]
#         new_w = " ".join(w)
#         words.append(new_w)
#     return words
#
# def add_new_column(labels, split, name="test_demo"):
#     path = "D:\\AI\\Graduation_Project\\model\\BATM\\dataset\\data\\" + name + ".csv"
#     list_index = [str(i) for i in range(0,768)]
#     df = pd.read_csv(path, encoding="utf-8",names=list_index)
#     data = []
#     data.append(list_index)
#     data.extend(np.array(df))
#     new_label = ["label"]
#     new_label.extend(labels)
#     new_split = ["split"]
#     new_split.extend(split)
#     data = np.append(data, [[num] for num in new_label], axis=1)
#     data = np.append(data, [[num] for num in new_split], axis=1)
#     new_df = pd.DataFrame(data)
#     # df = pd.concat([new_df,df],ignore_index=True,axis=0)
#     new_df.to_csv('D:\\AI\\Graduation_Project\\model\\BATM\\dataset\\data\\' + name + ".csv", index=False, header=False)
#
# def get_doc_embedding(name="test_demo"):
#     path = "D:\\AI\\Graduation_Project\\model\\BATM\\dataset\\data\\" + name + ".csv"
#     list_index = [str(i) for i in range(0, 768)]
#     list_index.append("label")
#     list_index.append("split")
#     df = pd.read_csv(path, encoding="utf-8")
#     # print(df["split"]=="valid")
#     print(df.iloc[:,:768])
#
# if __name__ == "__main__":
#     data_loader = get_read()
#     generate(data_loader,name="MIND15-Roberta", model="roberta-base")
    # add_new_column(data_loader["label"], data_loader["split"], "MIND15-DSBERT")
    # get_doc_embedding("test_demo")




# print("naive ...")
# keywords = kw_model.extract_keywords(doc)
# print(keywords)
#
# print("\nkeyphrase_ngram_range ...")
# keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)
# print(keywords)
#
# print("\nhighlight ...")
# keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), highlight=True)
# print(keywords)
#
# # 为了使结果多样化，我们将 2 x top_n 与文档最相似的词/短语。
# # 然后，我们从 2 x top_n 单词中取出所有 top_n 组合，并通过余弦相似度提取彼此最不相似的组合。
# print("\nuse_maxsum ...")

#
# # 为了使结果多样化，我们可以使用最大边界相关算法(MMR)
# # 来创建同样基于余弦相似度的关键字/关键短语。 具有高度多样性的结果：
# print("\nhight diversity ...")
# keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english',
#                         use_mmr=True, diversity=0.7)
# print(keywords)
#
#
# # 低多样性的结果
# print("\nlow diversity ...")
# keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english',
# #                         use_mmr=True, diversity=0.2)
# print(keywords)


# from keybert import KeyBERT
# import string
# import re
#
#
# def clean_text(text):
#     rule = string.punctuation + "0123456789"
#     return re.sub(rf'([^{rule}a-zA-Z ])', r" ", text)
#
#
# def clean_df(data_df):
#     # 这行代码的作用是从数据表 data_df 中删除那些在"标题" (title) 和 "正文" (body) 字段中同时为空的行。参数 inplace=True 表示这一操作直接就地对 data_df 进行修改。
#     data_df.dropna(subset=["title", "body"], inplace=True, how="all")
#     # 这行代码的作用是将 data_df 中的空值（NA 或 NaN）替换为字符串 "empty"。参数 inplace=True 表示这一操作直接在 data_df 上进行修改。
#     data_df.fillna("empty", inplace=True)
#     # 这行代码使用一个匿名函数（lambda 函数）去处理 data_df 中的 title 列。函数 clean_text(s) 应该是一个对字符串进行清洗的函数，即对每一篇文章的标题进行清洗。
#     data_df["title"] = data_df.title.apply(lambda s: clean_text(s))
#     data_df["body"] = data_df.body.apply(lambda s: clean_text(s))
#     return data_df
#
# def get_read(name="MIND15"):
#     df = clean_df(pd.read_csv("/kaggle/input/mind15-and-news26-and-glove/dataset/data/" + name + ".csv", encoding="utf-8"))
#     # data = [df.iloc[:, 2], df.iloc[:, 9], df.iloc[:, 10]]
#     # print(len(data))
#     # print(data)
#     df["data"] = df.title + "\n" + df.body
#     # labels = df["category"].values.tolist()
#     # label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
#     # labels = [label_dict[x] for x in labels]
#     data = {}
#     data["text"] = df["data"]
#     data["label"] = df["category"].values.tolist()
#     data["split"] = df.iloc[:, 10]
#     return data
#
# def concat_words(keyword):
#     words = []
#     for word in keyword:
#         w = [x[0] for x in word]
#         new_w = " ".join(w)
#         words.append(new_w)
#     return words
#
#
# def generate(data_loader,name="test_demo", model="distilbert-base-uncased", batch_size=512):
#     model = KeyBERT(model= "/kaggle/input/roberta/roberta-base")
#     result = [["text", "label", "split"]]
#     labels = data_loader["label"]
#     split = data_loader["split"]
#     text = data_loader["text"]
#     length = len(text)
#     assert len(split)==len(labels)==len(text)
#     count = 1
#     for i in range(0, length, batch_size):
#         end = min(length, i + batch_size)
#         data = list(text[i:end])
#         keywords = model.extract_keywords(data, keyphrase_ngram_range=(1, 2), stop_words="english")
#         keywords = concat_words(keywords)
#         temp = [[keywords[j], labels[i + j], split[i + j]] for j in range(len(keywords))]
#         result.extend(temp)
#         if (end >= count * 10000):
#             print(end)
#             count += 1
#     # print(result)
#     df = pd.DataFrame(result)
#     csv_dile_path = '/kaggle/working/' + name + '.csv'
#     df.to_csv(csv_dile_path, index=False, header=False)
#     print("CSV 文件已保存到:", csv_dile_path)

import pandas as pd


# def regular_text(pad,data,mode="train"):
#     for x in zip(pad.label, pad.body):
#         data.append([x[0],x[1],mode])
#     return data
# root = "D:\\AI\\Graduation_Project\\model\\dataset\\data\\cnews\\"
# train_data=pd.read_csv(root + 'cnews_train.txt',sep='\t',names=['label','body'])
# valid_data=pd.read_csv(root + 'cnews_val.txt',sep='\t',names=['label','body'])
# test_data=pd.read_csv(root + 'cnews_test.txt',sep='\t',names=['label','body'])
# data=[["label","body","split"]]
# data = regular_text(train_data, data, mode="train")
# data = regular_text(valid_data, data, mode="valid")
# data = regular_text(test_data, data, mode="test")
# npd = pd.DataFrame(data)
# npd.to_csv(root + "cnews.csv", index=False, header=False)


# # 实现文本的预处理，保存至csv文件
# def label_text(data):
#     items = data.split('_!_')
#     return items[2], items[3] + items[4]
# df = pd.DataFrame(columns=['label', 'text'])
# with open('D:\\AI\\Graduation_Project\\model\\dataset\\data\\toutiao-text-classfication-dataset-master\\toutiao_cat_data.txt', 'r', encoding='utf-8') as file:
#     count = 1
#     for line in file:
#         label, text = label_text(line)
#         df = df._append({'label': label, 'text': text}, ignore_index=True)
#         count+=1
#         if count % 10000 == 0:
#             print(count)
# df.to_csv('D:\\AI\\Graduation_Project\\model\\dataset\\data\\toutiao-text-classfication-dataset-master\\toutiao_cat_data.csv', index=False, header=True)



# def regular_text(labels, data, split_list,new_data):
#     for x in zip(labels, data, split_list):
#         new_data.append([x[0],x[1],x[2]])
#     return new_data
#
# category_list = ["news_entertainment","news_sports","news_finance","news_house","news_car","news_edu","news_tech","news_military","news_world","news_agriculture","news_game",]
# df = pd.read_csv("D:\\AI\\Graduation_Project\\model\\dataset\\data\\toutiao-text-classfication-dataset-master\\toutiao_cat_data.csv")
# new_data = [["label","body","split"]]
#
# split_list = ['train' for _ in range(10000)]
# split_list.extend(['valid' for _ in range(1000)])
# split_list.extend(['test' for _ in range(1000)])
# count = 0
# for category in category_list:
#     d = df["label"] == category
#     data = random.sample(list(df[d].text), 12000)
#     labels = [category for _ in range(12000)]
#     regular_text(labels, data, split_list,new_data)
#     count += 1
#     print(count)
# ndf = pd.DataFrame(new_data)
# ndf.to_csv('D:\\AI\\Graduation_Project\\model\\dataset\\data\\toutiao-text-classfication-dataset-master\\new_toutiao_cat_data.csv', index=False, header=False)


# cnews 919.5041384615384
# toutiao 39.93841666666667

# df_roberta = pd.read_csv("D:\\AI\\Graduation_Project\\model\\dataset\\data\\MIND15-Roberta-3.csv")
# df = pd.read_csv("D:\\AI\\Graduation_Project\\model\\dataset\\data\\MIND15.csv")
# df["title"] = df_roberta.text + "\n" + df.title
# df.to_csv("D:\\AI\\Graduation_Project\\model\\dataset\\data\\New_MIND15.csv")


def extract_topic(self, input_feat):
    embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
    if self.with_gru:
        length = torch.sum(input_feat["mask"], dim=-1)
        # x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
        output, attn_out, weight = self.rnn(embedding)
        topic_weight = self.droputout1(
            self.add_norm1(self.W_q(embedding).transpose(1, 2), self.W_k(output).transpose(1, 2)))  # (N, S, E)
    # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
    # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
    # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
    # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
    mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
    topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
    topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
    return topic_vec, topic_weight, attn_out


def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
    # 获取输入的 tokens, mask等数据
    input_feat["embedding"] = input_feat.get("embedding")
    # 嵌入层，对 tokens 向量化
    embedding = self.embedding_layer(input_feat)
    # 门控循环单元层。
    '''
    output:GRU每个时间步的输出； 
    attn_out: 通过注意力加权和的输出,也是初级文档表示向量；
    weight：注意力权重
    '''
    output, attn_out, weight = GRU(embedding)
    # 第一个注意力层，获取到主题-单词权重
    topic_weight = Linear(embedding).transpose(1, 2) + Linear(output).transpose(1, 2)
    # 通过主题-单词分布得到主题向量
    topic_vec = self.final(torch.matmul(topic_weight, embedding))
    # 第二个注意力层。
    '''
    doc_embedding: 文档表示向量
    doc_topic: 文档-主题权重
    '''
    doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
    # 与GRU的初级文档表示向量融合，得到最终的文档表示向量
    doc_embedding += attn_out
    # 分类层
    output = self.classify_layer(doc_embedding)
    return output






