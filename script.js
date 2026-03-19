/* ============================================================
   CineMatch AI – script.js  (Full ML Recommendation Engine)
   ============================================================ */

// ── MOVIE DATABASE ──────────────────────────────────────────
const MOVIES = [
  // HOLLYWOOD
  {id:1,title:"The Dark Knight",year:2008,genre:["Action","Crime","Drama"],industry:"Hollywood",language:"English",rating:9.0,director:"Christopher Nolan",cast:"Christian Bale, Heath Ledger",emoji:"🦇",streaming:["Netflix","Amazon Prime"],desc:"Batman faces the Joker in Gotham City."},
  {id:2,title:"Inception",year:2010,genre:["Sci-Fi","Thriller","Action"],industry:"Hollywood",language:"English",rating:8.8,director:"Christopher Nolan",cast:"Leonardo DiCaprio, Joseph Gordon-Levitt",emoji:"🌀",streaming:["Netflix","HBO Max"],desc:"A thief who enters dreams to steal secrets."},
  {id:3,title:"Interstellar",year:2014,genre:["Sci-Fi","Drama","Adventure"],industry:"Hollywood",language:"English",rating:8.6,director:"Christopher Nolan",cast:"Matthew McConaughey, Anne Hathaway",emoji:"🚀",streaming:["Netflix","Amazon Prime"],desc:"Astronauts travel through a wormhole near Saturn."},
  {id:4,title:"Avengers: Endgame",year:2019,genre:["Action","Fantasy","Sci-Fi"],industry:"Hollywood",language:"English",rating:8.4,director:"Russo Brothers",cast:"Robert Downey Jr., Chris Evans",emoji:"⚡",streaming:["Disney+ Hotstar"],desc:"The Avengers reassemble to reverse Thanos's actions."},
  {id:5,title:"The Shawshank Redemption",year:1994,genre:["Drama","Crime"],industry:"Hollywood",language:"English",rating:9.3,director:"Frank Darabont",cast:"Tim Robbins, Morgan Freeman",emoji:"🏛️",streaming:["Netflix","Amazon Prime"],desc:"Two imprisoned men bond over years of hardship."},
  {id:6,title:"The Godfather",year:1972,genre:["Crime","Drama"],industry:"Hollywood",language:"English",rating:9.2,director:"Francis Ford Coppola",cast:"Marlon Brando, Al Pacino",emoji:"🌹",streaming:["Amazon Prime"],desc:"The aging patriarch of an organized crime dynasty transfers control."},
  {id:7,title:"Pulp Fiction",year:1994,genre:["Crime","Drama","Thriller"],industry:"Hollywood",language:"English",rating:8.9,director:"Quentin Tarantino",cast:"John Travolta, Samuel L. Jackson",emoji:"🍔",streaming:["Netflix","Amazon Prime"],desc:"Interlocking tales of crime in Los Angeles."},
  {id:8,title:"The Matrix",year:1999,genre:["Sci-Fi","Action"],industry:"Hollywood",language:"English",rating:8.7,director:"Wachowski Sisters",cast:"Keanu Reeves, Laurence Fishburne",emoji:"💊",streaming:["Netflix","HBO Max"],desc:"A computer hacker discovers the true nature of reality."},
  {id:9,title:"Spider-Man: No Way Home",year:2021,genre:["Action","Fantasy","Sci-Fi"],industry:"Hollywood",language:"English",rating:8.2,director:"Jon Watts",cast:"Tom Holland, Zendaya",emoji:"🕷️",streaming:["Netflix"],desc:"Spider-Man seeks help from Doctor Strange, unleashing the multiverse."},
  {id:10,title:"Oppenheimer",year:2023,genre:["Drama","Biography","Thriller"],industry:"Hollywood",language:"English",rating:8.9,director:"Christopher Nolan",cast:"Cillian Murphy, Emily Blunt",emoji:"💣",streaming:["Amazon Prime","Apple TV+"],desc:"The story of J. Robert Oppenheimer and atomic bomb development."},
  {id:11,title:"Top Gun: Maverick",year:2022,genre:["Action","Drama"],industry:"Hollywood",language:"English",rating:8.3,director:"Joseph Kosinski",cast:"Tom Cruise, Jennifer Connelly",emoji:"✈️",streaming:["Netflix","Amazon Prime"],desc:"Maverick trains a new generation of Top Gun pilots."},
  {id:12,title:"Dune",year:2021,genre:["Sci-Fi","Action","Adventure"],industry:"Hollywood",language:"English",rating:8.0,director:"Denis Villeneuve",cast:"Timothée Chalamet, Zendaya",emoji:"🏜️",streaming:["HBO Max","Amazon Prime"],desc:"A noble family's heir journeys to a desert planet."},
  {id:13,title:"Joker",year:2019,genre:["Crime","Drama","Thriller"],industry:"Hollywood",language:"English",rating:8.4,director:"Todd Phillips",cast:"Joaquin Phoenix, Robert De Niro",emoji:"🃏",streaming:["Netflix","HBO Max"],desc:"The origin story of Arthur Fleck's transformation into the Joker."},
  {id:14,title:"Avatar: The Way of Water",year:2022,genre:["Sci-Fi","Fantasy","Action"],industry:"Hollywood",language:"English",rating:7.6,director:"James Cameron",cast:"Sam Worthington, Zoe Saldana",emoji:"🐳",streaming:["Disney+ Hotstar","Amazon Prime"],desc:"Jake and Neytiri defend their family on Pandora."},
  {id:15,title:"Titanic",year:1997,genre:["Romance","Drama","Action"],industry:"Hollywood",language:"English",rating:7.9,director:"James Cameron",cast:"Leonardo DiCaprio, Kate Winslet",emoji:"🚢",streaming:["Disney+ Hotstar","Netflix"],desc:"A young couple from different social classes fall in love aboard the Titanic."},
  {id:16,title:"Gladiator",year:2000,genre:["Action","Drama","Adventure"],industry:"Hollywood",language:"English",rating:8.5,director:"Ridley Scott",cast:"Russell Crowe, Joaquin Phoenix",emoji:"⚔️",streaming:["Amazon Prime","Netflix"],desc:"A Roman general seeks vengeance against the corrupt emperor."},
  {id:17,title:"Parasite",year:2019,genre:["Drama","Thriller","Comedy"],industry:"Korean",language:"Korean",rating:8.5,director:"Bong Joon-ho",cast:"Song Kang-ho, Lee Sun-kyun",emoji:"🏡",streaming:["Amazon Prime","Apple TV+"],desc:"A poor family schemes to become employed by a wealthy household."},
  {id:18,title:"Forrest Gump",year:1994,genre:["Drama","Romance","Comedy"],industry:"Hollywood",language:"English",rating:8.8,director:"Robert Zemeckis",cast:"Tom Hanks, Robin Wright",emoji:"🏃",streaming:["Amazon Prime"],desc:"The life journey of a man with a low IQ who witnesses historic events."},
  {id:19,title:"Black Panther",year:2018,genre:["Action","Fantasy","Sci-Fi"],industry:"Hollywood",language:"English",rating:7.3,director:"Ryan Coogler",cast:"Chadwick Boseman, Michael B. Jordan",emoji:"🐆",streaming:["Disney+ Hotstar"],desc:"T'Challa returns home to Wakanda after his father's death."},
  {id:20,title:"Barbie",year:2023,genre:["Comedy","Fantasy","Drama"],industry:"Hollywood",language:"English",rating:7.0,director:"Greta Gerwig",cast:"Margot Robbie, Ryan Gosling",emoji:"👗",streaming:["HBO Max","Amazon Prime"],desc:"Barbie and Ken travel from Barbie Land to the real world."},
  {id:21,title:"Gladiator II",year:2024,genre:["Action","Drama","Adventure"],industry:"Hollywood",language:"English",rating:7.2,director:"Ridley Scott",cast:"Paul Mescal, Denzel Washington",emoji:"🏛️",streaming:["Amazon Prime"],desc:"A young man fights in the Roman Colosseum for revenge and glory."},
  {id:22,title:"Deadpool & Wolverine",year:2024,genre:["Action","Comedy","Sci-Fi"],industry:"Hollywood",language:"English",rating:8.0,director:"Shawn Levy",cast:"Ryan Reynolds, Hugh Jackman",emoji:"🔴",streaming:["Disney+ Hotstar"],desc:"Deadpool and Wolverine team up in the Marvel multiverse."},

  // BOLLYWOOD
  {id:30,title:"3 Idiots",year:2009,genre:["Comedy","Drama"],industry:"Bollywood",language:"Hindi",rating:8.4,director:"Rajkumar Hirani",cast:"Aamir Khan, R. Madhavan, Sharman Joshi",emoji:"🎓",streaming:["Netflix","Amazon Prime"],desc:"Three engineering friends challenge the education system."},
  {id:31,title:"Dangal",year:2016,genre:["Drama","Sport","Biography"],industry:"Bollywood",language:"Hindi",rating:8.4,director:"Nitesh Tiwari",cast:"Aamir Khan, Fatima Sana Shaikh",emoji:"🤼",streaming:["Netflix","Disney+ Hotstar"],desc:"A wrestling coach trains his daughters to become champions."},
  {id:32,title:"PK",year:2014,genre:["Comedy","Drama","Sci-Fi"],industry:"Bollywood",language:"Hindi",rating:8.1,director:"Rajkumar Hirani",cast:"Aamir Khan, Anushka Sharma",emoji:"👽",streaming:["Netflix","Amazon Prime"],desc:"An alien stranded on Earth questions religious institutions."},
  {id:33,title:"Dilwale Dulhania Le Jayenge",year:1995,genre:["Romance","Drama","Comedy"],industry:"Bollywood",language:"Hindi",rating:8.1,director:"Aditya Chopra",cast:"Shah Rukh Khan, Kajol",emoji:"🌹",streaming:["Netflix"],desc:"Two Indians fall in love in Europe; a classic Bollywood romance."},
  {id:34,title:"Sholay",year:1975,genre:["Action","Crime","Drama"],industry:"Bollywood",language:"Hindi",rating:8.2,director:"Ramesh Sippy",cast:"Dharmendra, Amitabh Bachchan, Hema Malini",emoji:"🔫",streaming:["Amazon Prime","YouTube"],desc:"Two criminals are hired to capture a ruthless dacoit."},
  {id:35,title:"Lagaan",year:2001,genre:["Drama","Sport","History"],industry:"Bollywood",language:"Hindi",rating:8.1,director:"Ashutosh Gowariker",cast:"Aamir Khan, Gracy Singh",emoji:"🏏",streaming:["Amazon Prime"],desc:"Villagers challenge British rulers to a game of cricket."},
  {id:36,title:"Andhadhun",year:2018,genre:["Thriller","Crime","Comedy"],industry:"Bollywood",language:"Hindi",rating:8.3,director:"Sriram Raghavan",cast:"Ayushmann Khurrana, Tabu",emoji:"🎹",streaming:["Netflix"],desc:"A blind pianist accidentally witnesses a murder."},
  {id:37,title:"Gully Boy",year:2019,genre:["Drama","Music"],industry:"Bollywood",language:"Hindi",rating:7.9,director:"Zoya Akhtar",cast:"Ranveer Singh, Alia Bhatt",emoji:"🎤",streaming:["Amazon Prime","Netflix"],desc:"A street rapper from Mumbai slums rises to fame."},
  {id:38,title:"URI: The Surgical Strike",year:2019,genre:["Action","Drama","Biography"],industry:"Bollywood",language:"Hindi",rating:8.2,director:"Aditya Dhar",cast:"Vicky Kaushal, Paresh Rawal",emoji:"🎖️",streaming:["ZEE5","Amazon Prime"],desc:"India's surgical strike against terrorist camps in Pakistan."},
  {id:39,title:"RRR",year:2022,genre:["Action","Drama","Fantasy"],industry:"Bollywood",language:"Hindi",rating:7.9,director:"S. S. Rajamouli",cast:"Ram Charan, Jr. NTR",emoji:"🔥",streaming:["Netflix","ZEE5"],desc:"Two legendary freedom fighters and their journey in 1920s India."},
  {id:40,title:"KGF: Chapter 2",year:2022,genre:["Action","Crime","Drama"],industry:"Kannada",language:"Kannada",rating:8.2,director:"Prashanth Neel",cast:"Yash, Sanjay Dutt",emoji:"🏆",streaming:["Amazon Prime"],desc:"Rocky rises to become the dominant power in the Kolar Gold Fields."},
  {id:41,title:"Pathaan",year:2023,genre:["Action","Thriller"],industry:"Bollywood",language:"Hindi",rating:5.9,director:"Siddharth Anand",cast:"Shah Rukh Khan, Deepika Padukone",emoji:"💥",streaming:["Amazon Prime"],desc:"A spy returns from exile to stop a massive terrorist attack."},
  {id:42,title:"Jawan",year:2023,genre:["Action","Drama","Thriller"],industry:"Bollywood",language:"Hindi",rating:7.1,director:"Atlee",cast:"Shah Rukh Khan, Nayanthara",emoji:"🚨",streaming:["Netflix"],desc:"A prison warden orchestrates a series of crimes to send a message."},
  {id:43,title:"Zindagi Na Milegi Dobara",year:2011,genre:["Drama","Comedy","Romance"],industry:"Bollywood",language:"Hindi",rating:8.2,director:"Zoya Akhtar",cast:"Hrithik Roshan, Farhan Akhtar, Abhay Deol",emoji:"🌊",streaming:["Netflix"],desc:"Three friends go on a bachelor trip across Spain."},
  {id:44,title:"Queen",year:2014,genre:["Drama","Romance","Comedy"],industry:"Bollywood",language:"Hindi",rating:8.2,director:"Vikas Bahl",cast:"Kangana Ranaut, Rajkummar Rao",emoji:"👑",streaming:["Amazon Prime"],desc:"A girl goes on her honeymoon alone after her fiancé calls off the wedding."},
  {id:45,title:"Kabhi Khushi Kabhie Gham",year:2001,genre:["Drama","Romance","Comedy"],industry:"Bollywood",language:"Hindi",rating:7.4,director:"Karan Johar",cast:"Shah Rukh Khan, Amitabh Bachchan, Kajol",emoji:"🎊",streaming:["Amazon Prime","Netflix"],desc:"A rich family and the value of relationships and traditions."},
  {id:46,title:"Animal",year:2023,genre:["Action","Crime","Drama"],industry:"Bollywood",language:"Hindi",rating:7.0,director:"Sandeep Reddy Vanga",cast:"Ranbir Kapoor, Anil Kapoor",emoji:"🐺",streaming:["Netflix"],desc:"A man fiercely protects his family empire while battling dark impulses."},
  {id:47,title:"Dunki",year:2023,genre:["Comedy","Drama","Romance"],industry:"Bollywood",language:"Hindi",rating:5.9,director:"Rajkumar Hirani",cast:"Shah Rukh Khan, Taapsee Pannu",emoji:"🌍",streaming:["Netflix"],desc:"Friends attempt illegal immigration routes to reach their dream country."},

  // TOLLYWOOD
  {id:60,title:"Baahubali: The Beginning",year:2015,genre:["Action","Drama","Fantasy"],industry:"Tollywood",language:"Telugu",rating:8.0,director:"S. S. Rajamouli",cast:"Prabhas, Rana Daggubati",emoji:"👑",streaming:["Netflix","Amazon Prime"],desc:"A young man uncovers his epic destiny as a warrior prince."},
  {id:61,title:"Baahubali 2: The Conclusion",year:2017,genre:["Action","Drama","Fantasy"],industry:"Tollywood",language:"Telugu",rating:8.2,director:"S. S. Rajamouli",cast:"Prabhas, Anushka Shetty",emoji:"⚔️",streaming:["Netflix","Amazon Prime"],desc:"The secret of why Kattappa killed Baahubali is revealed."},
  {id:62,title:"RRR",year:2022,genre:["Action","Drama","Fantasy"],industry:"Tollywood",language:"Telugu",rating:8.0,director:"S. S. Rajamouli",cast:"Ram Charan, Jr. NTR",emoji:"🔥",streaming:["Netflix"],desc:"Two legendary revolutionaries rise against British colonizers."},
  {id:63,title:"Arjun Reddy",year:2017,genre:["Drama","Romance"],industry:"Tollywood",language:"Telugu",rating:8.1,director:"Sandeep Reddy Vanga",cast:"Vijay Deverakonda, Shalini Pandey",emoji:"💔",streaming:["Amazon Prime"],desc:"A self-destructive doctor spirals after a painful breakup."},
  {id:64,title:"Mahanati",year:2018,genre:["Drama","Biography","Romance"],industry:"Tollywood",language:"Telugu",rating:8.7,director:"Nag Ashwin",cast:"Keerthy Suresh, Dulquer Salmaan",emoji:"🎬",streaming:["Amazon Prime"],desc:"The life and tragic story of legendary actress Savitri."},
  {id:65,title:"Pushpa: The Rise",year:2021,genre:["Action","Crime","Drama"],industry:"Tollywood",language:"Telugu",rating:7.6,director:"Sukumar",cast:"Allu Arjun, Rashmika Mandanna",emoji:"🌿",streaming:["Amazon Prime"],desc:"A lorry driver rises through the red sandalwood smuggling syndicate."},
  {id:66,title:"Pushpa 2: The Rule",year:2024,genre:["Action","Crime","Drama"],industry:"Tollywood",language:"Telugu",rating:7.9,director:"Sukumar",cast:"Allu Arjun, Rashmika Mandanna",emoji:"🔱",streaming:["Amazon Prime"],desc:"Pushpa expands his empire and faces bigger threats."},
  {id:67,title:"Ala Vaikunthapurramuloo",year:2020,genre:["Action","Comedy","Drama"],industry:"Tollywood",language:"Telugu",rating:7.7,director:"Trivikram Srinivas",cast:"Allu Arjun, Pooja Hegde",emoji:"🏠",streaming:["Amazon Prime"],desc:"A man who was swapped at birth discovers his true identity."},
  {id:68,title:"Rangasthalam",year:2018,genre:["Drama","Action","Romance"],industry:"Tollywood",language:"Telugu",rating:8.3,director:"Sukumar",cast:"Ram Charan, Samantha Ruth Prabhu",emoji:"🌾",streaming:["Amazon Prime"],desc:"A partially deaf man fights against a corrupt village president."},
  {id:69,title:"Jersey",year:2019,genre:["Drama","Sport"],industry:"Tollywood",language:"Telugu",rating:8.6,director:"Gowtam Tinnanuri",cast:"Nani, Shraddha Srinath",emoji:"🏏",streaming:["Netflix"],desc:"A former cricketer makes a comeback to gift his son a jersey."},
  {id:70,title:"Uppena",year:2021,genre:["Romance","Drama","Tragedy"],industry:"Tollywood",language:"Telugu",rating:7.7,director:"Buchi Babu Sana",cast:"Panja Vaisshnav Tej, Krithi Shetty",emoji:"🌊",streaming:["ZEE5","Amazon Prime"],desc:"A young man falls for an affluent girl from a higher caste."},
  {id:71,title:"Shyam Singha Roy",year:2021,genre:["Drama","Romance","Mystery"],industry:"Tollywood",language:"Telugu",rating:7.9,director:"Rahul Sankrityan",cast:"Nani, Sai Pallavi",emoji:"✍️",streaming:["Netflix"],desc:"A film director discovers his past life connection to a reformer."},
  {id:72,title:"Vakeel Saab",year:2021,genre:["Drama","Crime"],industry:"Tollywood",language:"Telugu",rating:7.4,director:"Venu Sriram",cast:"Pawan Kalyan, Anjali",emoji:"⚖️",streaming:["Amazon Prime"],desc:"A lawyer defends three women who killed a sexual predator."},
  {id:73,title:"Kalki 2898 AD",year:2024,genre:["Sci-Fi","Action","Fantasy"],industry:"Tollywood",language:"Telugu",rating:6.8,director:"Nag Ashwin",cast:"Prabhas, Deepika Padukone, Amitabh Bachchan",emoji:"🌌",streaming:["Netflix"],desc:"In a dystopian future, a bounty hunter and a woman carry destiny."},
  {id:74,title:"Devara",year:2024,genre:["Action","Thriller","Drama"],industry:"Tollywood",language:"Telugu",rating:6.5,director:"Koratala Siva",cast:"Jr. NTR, Saif Ali Khan",emoji:"🌊",streaming:["Netflix"],desc:"A feared coastal ruler's son must rise to his legacy."},
  {id:75,title:"Maharshi",year:2019,genre:["Drama","Romance"],industry:"Tollywood",language:"Telugu",rating:7.3,director:"Vamshi Paidipally",cast:"Mahesh Babu, Pooja Hegde",emoji:"🌱",streaming:["Amazon Prime"],desc:"A successful businessman returns to his roots to help farmers."},
  {id:76,title:"Sarkaru Vaari Paata",year:2022,genre:["Action","Comedy","Drama"],industry:"Tollywood",language:"Telugu",rating:6.9,director:"Parasuram",cast:"Mahesh Babu, Keerthy Suresh",emoji:"💰",streaming:["Amazon Prime"],desc:"A loan recovery officer takes revenge on a corrupt bank."},
  {id:77,title:"Bheemla Nayak",year:2022,genre:["Action","Drama"],industry:"Tollywood",language:"Telugu",rating:7.7,director:"Saagar K Chandra",cast:"Pawan Kalyan, Rana Daggubati",emoji:"🏔️",streaming:["Amazon Prime"],desc:"Two strong personalities clash over ego and principles."},

  // KOLLYWOOD
  {id:90,title:"Vikram",year:2022,genre:["Action","Thriller","Crime"],industry:"Kollywood",language:"Tamil",rating:8.4,director:"Lokesh Kanagaraj",cast:"Kamal Haasan, Vijay Sethupathi, Fahadh Faasil",emoji:"💀",streaming:["Amazon Prime"],desc:"A retired special agent investigates a series of masked murders."},
  {id:91,title:"Jailer",year:2023,genre:["Action","Drama","Crime"],industry:"Kollywood",language:"Tamil",rating:7.1,director:"Nelson Dilipkumar",cast:"Rajinikanth, Mohanlal, Jackie Shroff",emoji:"🚔",streaming:["Amazon Prime"],desc:"A retired jailer goes after the gang that kidnapped his son."},
  {id:92,title:"Leo",year:2023,genre:["Action","Thriller","Crime"],industry:"Kollywood",language:"Tamil",rating:7.4,director:"Lokesh Kanagaraj",cast:"Vijay, Trisha, Sanjay Dutt",emoji:"🦁",streaming:["Netflix"],desc:"A gentle café owner is forced to confront his violent past."},
  {id:93,title:"Ponniyin Selvan I",year:2022,genre:["Drama","History","Action"],industry:"Kollywood",language:"Tamil",rating:7.6,director:"Mani Ratnam",cast:"Vikram, Karthi, Jayam Ravi",emoji:"👒",streaming:["Amazon Prime"],desc:"A prince navigates conspiracies within the Chola Empire."},
  {id:94,title:"Ponniyin Selvan II",year:2023,genre:["Drama","History","Action"],industry:"Kollywood",language:"Tamil",rating:7.2,director:"Mani Ratnam",cast:"Vikram, Aishwarya Rai",emoji:"⚜️",streaming:["Amazon Prime"],desc:"The Chola succession war reaches its dramatic climax."},
  {id:95,title:"Master",year:2021,genre:["Action","Thriller","Drama"],industry:"Kollywood",language:"Tamil",rating:7.2,director:"Lokesh Kanagaraj",cast:"Vijay, Vijay Sethupathi",emoji:"🏫",streaming:["Amazon Prime","Netflix"],desc:"An alcoholic professor battles a crime lord at a juvenile school."},
  {id:96,title:"Kaithi",year:2019,genre:["Action","Thriller"],industry:"Kollywood",language:"Tamil",rating:8.5,director:"Lokesh Kanagaraj",cast:"Karthi, Narain",emoji:"🌃",streaming:["Amazon Prime"],desc:"An ex-convict must transport police officers through a dangerous night."},
  {id:97,title:"Soorarai Pottru",year:2020,genre:["Drama","Biography","Sport"],industry:"Kollywood",language:"Tamil",rating:8.7,director:"Sudha Kongara",cast:"Suriya, Aparna Balamurali",emoji:"✈️",streaming:["Amazon Prime"],desc:"The true story behind a low-cost airline in India."},
  {id:98,title:"96",year:2018,genre:["Romance","Drama"],industry:"Kollywood",language:"Tamil",rating:8.5,director:"C. Prem Kumar",cast:"Vijay Sethupathi, Trisha",emoji:"💌",streaming:["SonyLIV","Amazon Prime"],desc:"Two classmates reunite after 22 years and reminisce about first love."},
  {id:99,title:"Mersal",year:2017,genre:["Action","Drama","Crime"],industry:"Kollywood",language:"Tamil",rating:7.7,director:"Atlee",cast:"Vijay, Samantha Ruth Prabhu",emoji:"💉",streaming:["ZEE5","Amazon Prime"],desc:"A doctor fights corruption in the medical system."},
  {id:100,title:"Thuppakki",year:2012,genre:["Action","Thriller","Crime"],industry:"Kollywood",language:"Tamil",rating:8.2,director:"A. R. Murugadoss",cast:"Vijay, Kajal Aggarwal",emoji:"🔫",streaming:["ZEE5","Amazon Prime"],desc:"A military intelligence officer busts a sleeper terrorist cell."},
  {id:101,title:"Vettaiyan",year:2024,genre:["Action","Crime","Drama"],industry:"Kollywood",language:"Tamil",rating:6.9,director:"T. J. Gnanavel",cast:"Rajinikanth, Amitabh Bachchan, Fahadh Faasil",emoji:"🎯",streaming:["Netflix"],desc:"A senior police officer clashes with a righteous officer over justice."},

  // MOLLYWOOD
  {id:110,title:"Drishyam",year:2013,genre:["Thriller","Drama","Crime"],industry:"Mollywood",language:"Malayalam",rating:8.9,director:"Jeethu Joseph",cast:"Mohanlal, Meena",emoji:"📺",streaming:["Amazon Prime"],desc:"A man uses his movie knowledge to protect his family from a crime."},
  {id:111,title:"Kumbalangi Nights",year:2019,genre:["Drama","Romance","Comedy"],industry:"Mollywood",language:"Malayalam",rating:8.5,director:"Madhu C. Narayanan",cast:"Fahadh Faasil, Soubin Shahir",emoji:"🌅",streaming:["Amazon Prime"],desc:"Four estranged brothers and their dysfunctional family find redemption."},
  {id:112,title:"Minnal Murali",year:2021,genre:["Fantasy","Action","Comedy"],industry:"Mollywood",language:"Malayalam",rating:7.7,director:"Basil Paulose",cast:"Tovino Thomas, Guru Somasundaram",emoji:"⚡",streaming:["Netflix"],desc:"A local tailor gains superpowers and must battle a villain."},
  {id:113,title:"Hridayam",year:2022,genre:["Romance","Drama","Comedy"],industry:"Mollywood",language:"Malayalam",rating:8.3,director:"Vineeth Sreenivasan",cast:"Pranav Mohanlal, Darshana Rajendran",emoji:"❤️",streaming:["Disney+ Hotstar"],desc:"A young man's life and loves across college and adulthood."},
  {id:114,title:"2018",year:2023,genre:["Drama","Disaster","Biography"],industry:"Mollywood",language:"Malayalam",rating:8.8,director:"Jude Anthany Joseph",cast:"Tovino Thomas, Kunchacko Boban",emoji:"🌊",streaming:["Amazon Prime"],desc:"The harrowing 2018 Kerala floods and the rescue operations."},
  {id:115,title:"Manjummel Boys",year:2024,genre:["Adventure","Drama","Thriller"],industry:"Mollywood",language:"Malayalam",rating:8.5,director:"Chidambaram",cast:"Soubin Shahir, Sreenath Bhasi",emoji:"🧗",streaming:["Amazon Prime"],desc:"A group of friends attempt a daring cave rescue in the Himalayas."},
  {id:116,title:"Jallikattu",year:2019,genre:["Drama","Thriller","Action"],industry:"Mollywood",language:"Malayalam",rating:7.8,director:"Lijo Jose Pellissery",cast:"Antony Varghese, Chemban Vinod Jose",emoji:"🐂",streaming:["Amazon Prime"],desc:"A village descends into chaos trying to catch a rampaging bull."},
  {id:117,title:"Bheeshma Parvam",year:2022,genre:["Action","Crime","Drama"],industry:"Mollywood",language:"Malayalam",rating:7.7,director:"Sajid Yahiya",cast:"Mammootty, Shanvi Srivastava",emoji:"🦅",streaming:["Amazon Prime"],desc:"The patriarch of a family is pulled back into a power struggle."},

  // KOREAN
  {id:120,title:"Train to Busan",year:2016,genre:["Horror","Action","Thriller"],industry:"Korean",language:"Korean",rating:7.6,director:"Yeon Sang-ho",cast:"Gong Yoo, Ma Dong-seok",emoji:"🚆",streaming:["Netflix","Amazon Prime"],desc:"Passengers fight zombie hordes on a speeding train."},
  {id:121,title:"Oldboy",year:2003,genre:["Thriller","Mystery","Drama"],industry:"Korean",language:"Korean",rating:8.4,director:"Park Chan-wook",cast:"Choi Min-sik, Yoo Ji-tae",emoji:"🔍",streaming:["Amazon Prime"],desc:"A man imprisoned for 15 years seeks revenge after his release."},
  {id:122,title:"The Handmaiden",year:2016,genre:["Thriller","Drama","Romance"],industry:"Korean",language:"Korean",rating:8.1,director:"Park Chan-wook",cast:"Kim Min-hee, Ha Jung-woo",emoji:"🎭",streaming:["Amazon Prime"],desc:"A Japanese heiress is manipulated by a conman and a pickpocket."},

  // ANIME & JAPANESE
  {id:130,title:"Your Name",year:2016,genre:["Animation","Drama","Romance"],industry:"Japanese",language:"Japanese",rating:8.4,director:"Makoto Shinkai",cast:"Ryunosuke Kamiki",emoji:"☄️",streaming:["Netflix"],desc:"Two teenagers share a profound, magical connection upon discovering they are swapping bodies."},
  {id:131,title:"Spirited Away",year:2001,genre:["Animation","Adventure","Family"],industry:"Japanese",language:"Japanese",rating:8.6,director:"Hayao Miyazaki",cast:"Rumi Hiiragi",emoji:"🏮",streaming:["Netflix"],desc:"A young girl wanders into a world ruled by gods, witches, and spirits."},
  {id:132,title:"Seven Samurai",year:1954,genre:["Action","Drama"],industry:"Japanese",language:"Japanese",rating:8.6,director:"Akira Kurosawa",cast:"Toshiro Mifune",emoji:"⚔️",streaming:["Amazon Prime"],desc:"A poor village under attack by bandits recruits seven unemployed samurai to help them defend themselves."},
  {id:133,title:"Demon Slayer: Mugen Train",year:2020,genre:["Animation","Action","Fantasy"],industry:"Japanese",language:"Japanese",rating:8.2,director:"Haruo Sotozaki",cast:"Natsuki Hanae",emoji:"🗡️",streaming:["Netflix","Crunchyroll"],desc:"Tanjiro and his companions join the Flame Hashira Kyojuro Rengoku on a train."},

  // GLOBAL CINEMA
  {id:140,title:"Amélie",year:2001,genre:["Comedy","Romance"],industry:"French",language:"French",rating:8.3,director:"Jean-Pierre Jeunet",cast:"Audrey Tautou",emoji:"🥖",streaming:["Amazon Prime"],desc:"Amélie is an innocent and naive girl in Paris who decides to help those around her."},
  {id:141,title:"The Intouchables",year:2011,genre:["Comedy","Drama","Biography"],industry:"French",language:"French",rating:8.5,director:"Olivier Nakache",cast:"François Cluzet, Omar Sy",emoji:"♿",streaming:["Amazon Prime"],desc:"A billionaire becomes a quadriplegic and hires a young man from the projects to be his caregiver."},
  {id:142,title:"Pan's Labyrinth",year:2006,genre:["Fantasy","Drama","War"],industry:"Spanish",language:"Spanish",rating:8.2,director:"Guillermo del Toro",cast:"Ivana Baquero",emoji:"🧚",streaming:["Netflix"],desc:"In the Falangist Spain of 1944, the young stepdaughter of a sadistic army officer escapes into an eerie but captivating fantasy world."},
  {id:143,title:"Roma",year:2018,genre:["Drama"],industry:"Spanish",language:"Spanish",rating:7.7,director:"Alfonso Cuarón",cast:"Yalitza Aparicio",emoji:"🏠",streaming:["Netflix"],desc:"A year in the life of a middle-class family's maid in Mexico City in the early 1970s."},

  // ADDITIONAL INDIAN CINEMA
  {id:150,title:"Kantara",year:2022,genre:["Action","Drama","Thriller"],industry:"Kannada",language:"Kannada",rating:8.2,director:"Rishab Shetty",cast:"Rishab Shetty, Sapthami Gowda",emoji:"🔱",streaming:["Netflix","Amazon Prime"],desc:"A conflict between humanity and nature in a forest."},
  {id:151,title:"777 Charlie",year:2022,genre:["Adventure","Drama","Comedy"],industry:"Kannada",language:"Kannada",rating:8.7,director:"Kiranraj K",cast:"Rakshit Shetty",emoji:"🐕",streaming:["Amazon Prime"],desc:"Dharma is stuck in a rut with his lonely lifestyle and spends each day in the comfort of his loneliness."},
  {id:152,title:"Vikant Rona",year:2022,genre:["Action","Adventure","Mystery"],industry:"Kannada",language:"Kannada",rating:7.0,director:"Anup Bhandari",cast:"Kichcha Sudeepa",emoji:"🕵️",streaming:["ZEE5"],desc:"An inspector investigates a series of mysterious events in a remote village."},
  {id:153,title:"Saripodhaa Sanivaaram",year:2024,genre:["Action","Thriller"],industry:"Tollywood",language:"Telugu",rating:7.5,director:"Vivek Athreya",cast:"Nani, Priyanka Mohan",emoji:"🗓️",streaming:["Netflix"],desc:"A man who channels his anger only on Saturdays faces a ruthless cop."},
  {id:154,title:"The Goat Life",year:2024,genre:["Drama","Adventure"],industry:"Mollywood",language:"Malayalam",rating:8.1,director:"Blessy",cast:"Prithviraj Sukumaran",emoji:"🐐",streaming:["Netflix"],desc:"The true story of an Indian migrant worker who goes missing in Saudi Arabia."},
  {id:155,title:"Bramayugam",year:2024,genre:["Horror","Thriller"],industry:"Mollywood",language:"Malayalam",rating:7.9,director:"Rahul Sadasivan",cast:"Mammootty",emoji:"🏚️",streaming:["SonyLIV"],desc:"A folksinger narrowly escapes a slave market and finds himself in a mysterious mansion."},
  
  // HOLLYWOOD EXPANSION
  {id:160,title:"The Batman",year:2022,genre:["Action","Crime","Drama"],industry:"Hollywood",language:"English",rating:7.8,director:"Matt Reeves",cast:"Robert Pattinson",emoji:"🦇",streaming:["HBO Max"],desc:"Batman unearths corruption in Gotham City while chasing the Riddler."},
  {id:161,title:"Everything Everywhere All at Once",year:2022,genre:["Sci-Fi","Action","Comedy"],industry:"Hollywood",language:"English",rating:7.8,director:"Daniels",cast:"Michelle Yeoh",emoji:"🥯",streaming:["Hulu"],desc:"A middle-aged Chinese immigrant is swept up into an insane adventure."},
  {id:162,title:"The Whale",year:2022,genre:["Drama"],industry:"Hollywood",language:"English",rating:7.7,director:"Darren Aronofsky",cast:"Brendan Fraser",emoji:"🐋",streaming:["Apple TV+"],desc:"A reclusive, morbidly obese English teacher attempts to reconnect with his estranged teenage daughter."},
  {id:163,title:"Spider-Man: Across the Spider-Verse",year:2023,genre:["Animation","Action","Adventure"],industry:"Hollywood",language:"English",rating:8.6,director:"Joaquim Dos Santos",cast:"Shameik Moore",emoji:"🕸️",streaming:["Netflix"],desc:"Miles Morales catapults across the Multiverse."},
  {id:164,title:"John Wick: Chapter 4",year:2023,genre:["Action","Crime","Thriller"],industry:"Hollywood",language:"English",rating:7.7,director:"Chad Stahelski",cast:"Keanu Reeves",emoji:"🔫",streaming:["Amazon Prime"],desc:"John Wick uncovers a path to defeating the High Table."},
  {id:165,title:"Gladiator II",year:2024,genre:["Action","Drama","Adventure"],industry:"Hollywood",language:"English",rating:6.8,director:"Ridley Scott",cast:"Paul Mescal",emoji:"⚔️",streaming:["Paramount+"],desc:"Lucius, the grandson of Marcus Aurelius, enters the Colosseum."},

  // BOLLYWOOD EXPANSION
  {id:170,title:"Stree 2",year:2024,genre:["Horror","Comedy"],industry:"Bollywood",language:"Hindi",rating:7.6,director:"Amar Kaushik",cast:"Shraddha Kapoor, Rajkummar Rao",emoji:"👻",streaming:["Amazon Prime"],desc:"Chanderi is being haunted by a headless man Sarkata."},
  {id:171,title:"Fighter",year:2024,genre:["Action","Drama"],industry:"Bollywood",language:"Hindi",rating:6.8,director:"Siddharth Anand",cast:"Hrithik Roshan, Deepika Padukone",emoji:"🛩️",streaming:["Netflix"],desc:"Top IAF aviators come together in the face of imminent danger."},
  {id:172,title:"Munjya",year:2024,genre:["Horror","Comedy"],industry:"Bollywood",language:"Hindi",rating:7.1,director:"Aditya Sarpotdar",cast:"Abhay Verma",emoji:"👺",streaming:["Netflix"],desc:"A young man's visit to his native village unveils a dark family secret."},
  {id:173,title:"Crew",year:2024,genre:["Comedy","Crime"],industry:"Bollywood",language:"Hindi",rating:6.5,director:"Rajesh A Krishnan",cast:"Tabu, Kareena Kapoor Khan, Kriti Sanon",emoji:"✈️",streaming:["Netflix"],desc:"Three hard-working cabin crew members' lives lead to unexpected situations."},
  {id:174,title:"Shaitaan",year:2024,genre:["Horror","Thriller"],industry:"Bollywood",language:"Hindi",rating:7.0,director:"Vikas Bahl",cast:"Ajay Devgn, R. Madhavan",emoji:"🔥",streaming:["Netflix"],desc:"A family's vacation turns into a nightmare when they meet a stranger."},

  // TOLLYWOOD EXPANSION
  {id:180,title:"Lucky Baskhar",year:2024,genre:["Drama","Crime"],industry:"Tollywood",language:"Telugu",rating:8.2,director:"Venky Atluri",cast:"Dulquer Salmaan",emoji:"💰",streaming:["Netflix"],desc:"A bank cashier creates an extraordinary fortune through illegal means."},
  {id:181,title:"Guntur Kaaram",year:2024,genre:["Action","Drama"],industry:"Tollywood",language:"Telugu",rating:5.8,director:"Trivikram Srinivas",cast:"Mahesh Babu",emoji:"🌶️",streaming:["Netflix"],desc:"A rowdy from Guntur falls in love and rediscovers his family legacy."},
  {id:182,title:"Hanu-Man",year:2024,genre:["Fantasy","Action","Adventure"],industry:"Tollywood",language:"Telugu",rating:8.6,director:"Prasanth Varma",cast:"Teja Sajja",emoji:"🐒",streaming:["ZEE5"],desc:"A young man gets the superpowers of Lord Hanuman."},
  {id:183,title:"The Greatest of All Time",year:2024,genre:["Action","Sci-Fi","Thriller"],industry:"Kollywood",language:"Tamil",rating:6.4,director:"Venkat Prabhu",cast:"Vijay",emoji:"🐐",streaming:["Netflix"],desc:"A former field agent from the SATS team travels to Bangkok for a mission."},
  {id:184,title:"Raayan",year:2024,genre:["Action","Crime","Drama"],industry:"Kollywood",language:"Tamil",rating:6.9,director:"Dhanush",cast:"Dhanush",emoji:"🗡️",streaming:["Amazon Prime"],desc:"A common man is forced to protect his family and take revenge."},

  // KOREAN & ANIME EXPANSION
  {id:190,title:"Squid Game",year:2021,genre:["Thriller","Drama","Action"],industry:"Korean",language:"Korean",rating:8.0,director:"Hwang Dong-hyuk",cast:"Lee Jung-jae",emoji:"⭕",streaming:["Netflix"],desc:"Hundreds of cash-strapped contestants accept an invitation to compete in children's games."},
  {id:191,title:"Kingdom",year:2019,genre:["Horror","Action","History"],industry:"Korean",language:"Korean",rating:8.3,director:"Kim Seong-hun",cast:"Ju Ji-hoon",emoji:"🧟",streaming:["Netflix"],desc:"In the Joseon period, a crown prince investigates a mysterious plague."},
  {id:192,title:"All of Us Are Dead",year:2022,genre:["Horror","Action","Drama"],industry:"Korean",language:"Korean",rating:7.5,director:"Lee J.Q.",cast:"Park Ji-hu",emoji:"🏫",streaming:["Netflix"],desc:"High school students become trapped in their school during a zombie outbreak."},
  {id:193,title:"Spirited Away",year:2001,genre:["Animation","Adventure","Family"],industry:"Japanese",language:"Japanese",rating:8.6,director:"Hayao Miyazaki",cast:"Rumi Hiiragi",emoji:"🐉",streaming:["Netflix"],desc:"A ten-year-old girl wanders into a world ruled by gods, witches and spirits."},
  {id:194,title:"Your Name",year:2016,genre:["Animation","Romance","Fantasy"],industry:"Japanese",language:"Japanese",rating:8.4,director:"Makoto Shinkai",cast:"Ryunosuke Kamiki",emoji:"☄️",streaming:["Netflix"],desc:"Two teenagers share a profound, magical connection upon discovering they are swapping bodies."},
  {id:195,title:" Suzume",year:2022,genre:["Animation","Adventure","Fantasy"],industry:"Japanese",language:"Japanese",rating:7.7,director:"Makoto Shinkai",cast:"Nanoka Hara",emoji:"🚪",streaming:["Netflix"],desc:"A 17-year-old girl named Suzume helps a mysterious young man close doors from another world."},
];

// ── STREAMING BADGE MAP ─────────────────────────────────────
const STREAM_CLASS = {
  "Netflix":"sb-netflix","Amazon Prime":"sb-amazon",
  "Disney+ Hotstar":"sb-disney","ZEE5":"sb-zee5",
  "SonyLIV":"sb-sony","Apple TV+":"sb-apple",
  "HBO Max":"sb-hbo","YouTube":"sb-youtube","Hulu":"sb-hulu"
};
function streamBadgesHTML(arr){
  return arr.map(s=>`<span class="sb ${STREAM_CLASS[s]||''}">${s}</span>`).join('');
}

// ── INDUSTRY CSS CLASS MAP ──────────────────────────────────
function indClass(ind){
  return ({Hollywood:"hollywood",Bollywood:"bollywood",Tollywood:"tollywood",
    Kollywood:"kollywood",Mollywood:"mollywood",Korean:"korean",Kannada:"kannada"})[ind]||"hollywood";
}

// ── NAVBAR SCROLL ───────────────────────────────────────────
window.addEventListener('scroll',()=>document.getElementById('navbar').classList.toggle('scrolled',scrollY>40));

// ── HAMBURGER ───────────────────────────────────────────────
document.getElementById('hamburger').addEventListener('click',()=>document.getElementById('navLinks').classList.toggle('open'));
document.querySelectorAll('.nav-links a').forEach(a=>a.addEventListener('click',()=>document.getElementById('navLinks').classList.remove('open')));

// ── PARTICLES ───────────────────────────────────────────────
(()=>{const c=document.getElementById('particles');for(let i=0;i<50;i++){const p=document.createElement('div');p.className='particle';p.style.left=Math.random()*100+'%';p.style.top=Math.random()*100+'%';p.style.setProperty('--dur',(2+Math.random()*6)+'s');p.style.setProperty('--delay',(Math.random()*4)+'s');c.appendChild(p);}})();

// ── MOVIE EXPLORER ──────────────────────────────────────────
let currentMovies=[...MOVIES];

function getFilters(){
  return{
    search:document.getElementById('searchInput').value.trim().toLowerCase(),
    industry:document.getElementById('filterIndustry').value,
    language:document.getElementById('filterLanguage').value,
    genre:document.getElementById('filterGenre').value,
    year:document.getElementById('filterYear').value,
    streaming:document.getElementById('filterStreaming').value,
    sort:document.getElementById('sortBy').value,
  };
}

function applyFilters(){
  const f=getFilters();
  let list=[...MOVIES];

  if(f.search) list=list.filter(m=>
    m.title.toLowerCase().includes(f.search)||
    m.director.toLowerCase().includes(f.search)||
    m.cast.toLowerCase().includes(f.search)
  );
  if(f.industry!=='all') list=list.filter(m=>m.industry===f.industry);
  if(f.language!=='all') list=list.filter(m=>m.language===f.language);
  if(f.genre!=='all') list=list.filter(m=>m.genre.includes(f.genre));
  if(f.streaming!=='all') list=list.filter(m=>m.streaming.includes(f.streaming));
  if(f.year!=='all'){
    if(f.year==='pre2000') list=list.filter(m=>m.year<2000);
    else if(f.year.includes('-')){const[a,b]=f.year.split('-').map(Number);list=list.filter(m=>m.year>=a&&m.year<=b);}
    else list=list.filter(m=>m.year===parseInt(f.year));
  }

  // Sort
  if(f.sort==='rating') list.sort((a,b)=>b.rating-a.rating);
  else if(f.sort==='year-desc') list.sort((a,b)=>b.year-a.year);
  else if(f.sort==='year-asc') list.sort((a,b)=>a.year-b.year);
  else if(f.sort==='title') list.sort((a,b)=>a.title.localeCompare(b.title));

  currentMovies=list;
  renderMovies(list);
}

function renderMovies(list){
  const grid=document.getElementById('movieGridMain');
  const noRes=document.getElementById('noResults');
  const count=document.getElementById('movieCount');

  count.innerHTML=`Showing <strong>${list.length}</strong> of <strong>${MOVIES.length}</strong> movies`;

  if(list.length===0){grid.innerHTML='';noRes.style.display='block';return;}
  noRes.style.display='none';

  grid.innerHTML=list.map((m,i)=>{
    const ic=indClass(m.industry);
    const stars='⭐'.repeat(Math.round(m.rating/2));
    return `<div class="mc" onclick="openModal(${m.id})" style="animation-delay:${i*0.04}s">
      <div class="mc-poster poster-${ic.toLowerCase()}">
        <span class="mc-ind-badge ind-${ic}">${m.industry}</span>
        <span class="mc-year-badge">${m.year}</span>
        <div class="mc-stream-badges">${streamBadgesHTML(m.streaming)}</div>
        <span style="font-size:3rem">${m.emoji}</span>
      </div>
      <div class="mc-body">
        <div class="mc-title" title="${m.title}">${m.title}</div>
        <div class="mc-meta">
          <span class="mc-lang">${m.language}</span>·
          <span class="mc-rating">⭐ ${m.rating}</span>·
          <span>${m.year}</span>
        </div>
        <div class="mc-genres">${m.genre.map(g=>`<span class="mc-genre-tag">${g}</span>`).join('')}</div>
        <div class="mc-footer">
          <span class="mc-director" title="${m.director}">🎥 ${m.director}</span>
          <button class="mc-rec-btn" onclick="event.stopPropagation();quickRec(${m.id})">🤖 Similar</button>
        </div>
      </div>
    </div>`;
  }).join('');
}

// Filters event listeners
['searchInput','filterIndustry','filterLanguage','filterGenre','filterYear','filterStreaming','sortBy'].forEach(id=>
  document.getElementById(id).addEventListener('input',applyFilters)
);
document.getElementById('filterIndustry').addEventListener('change',applyFilters);
document.getElementById('filterLanguage').addEventListener('change',applyFilters);
document.getElementById('filterGenre').addEventListener('change',applyFilters);
document.getElementById('filterYear').addEventListener('change',applyFilters);
document.getElementById('filterStreaming').addEventListener('change',applyFilters);
document.getElementById('sortBy').addEventListener('change',applyFilters);

document.getElementById('searchInput').addEventListener('input',function(){
  document.getElementById('clearSearch').style.display=this.value?'block':'none';
});
document.getElementById('clearSearch').addEventListener('click',()=>{
  document.getElementById('searchInput').value='';
  document.getElementById('clearSearch').style.display='none';
  applyFilters();
});
document.getElementById('resetFilters').addEventListener('click',()=>{
  ['filterIndustry','filterLanguage','filterGenre','filterYear','filterStreaming'].forEach(id=>document.getElementById(id).value='all');
  document.getElementById('sortBy').value='rating';
  document.getElementById('searchInput').value='';
  document.getElementById('clearSearch').style.display='none';
  applyFilters();
});

// ── MODAL ───────────────────────────────────────────────────
function openModal(id){
  const m=MOVIES.find(x=>x.id===id);if(!m)return;
  const ic=indClass(m.industry);
  document.getElementById('modalContent').innerHTML=`
    <div class="modal-poster poster-${ic}" style="font-size:5rem">${m.emoji}
      <span class="mc-ind-badge ind-${ic}" style="position:absolute;top:14px;left:14px">${m.industry}</span>
    </div>
    <div class="modal-info">
      <h2>${m.title} (${m.year})</h2>
      <div class="modal-badges">
        <span class="mc-lang" style="font-size:.85rem;font-weight:700;color:var(--cyan)">🗣️ ${m.language}</span>
        <span style="color:var(--accent);font-weight:700">⭐ ${m.rating}/10</span>
      </div>
      <div class="modal-field"><span class="modal-field-label">📖 Story</span><span class="modal-field-val">${m.desc}</span></div>
      <div class="modal-field"><span class="modal-field-label">🎭 Genres</span><span class="modal-field-val">${m.genre.join(', ')}</span></div>
      <div class="modal-field"><span class="modal-field-label">🎥 Director</span><span class="modal-field-val">${m.director}</span></div>
      <div class="modal-field"><span class="modal-field-label">🎬 Cast</span><span class="modal-field-val">${m.cast}</span></div>
      <div class="modal-field"><span class="modal-field-label">📺 Streaming</span><span class="modal-field-val"><div class="modal-streams">${streamBadgesHTML(m.streaming)}</div></span></div>
      <button class="btn btn-primary" style="width:100%;margin-top:16px" onclick="quickRec(${m.id});document.getElementById('modalOverlay').classList.remove('open')">
        <span>Get AI Recommendations</span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
      </button>
    </div>`;
  document.getElementById('modalOverlay').classList.add('open');
}
document.getElementById('modalClose').addEventListener('click',()=>document.getElementById('modalOverlay').classList.remove('open'));
document.getElementById('modalOverlay').addEventListener('click',e=>{if(e.target===e.currentTarget)e.currentTarget.classList.remove('open');});

// ── ML RECOMMENDATION ENGINE ─────────────────────────────────
let selectedMovieId=null;

// Rec search
const recInput=document.getElementById('recSearchInput');
const recDrop=document.getElementById('recDropdown');

recInput.addEventListener('input',function(){
  const q=this.value.trim().toLowerCase();
  if(!q){recDrop.classList.remove('open');return;}
  const matches=MOVIES.filter(m=>m.title.toLowerCase().includes(q)).slice(0,8);
  if(!matches.length){recDrop.classList.remove('open');return;}
  recDrop.innerHTML=matches.map(m=>`
    <div class="rec-drop-item" onclick="selectMovieForRec(${m.id})">
      <span style="font-size:1.3rem">${m.emoji}</span>
      <span style="flex:1">${m.title} <span style="color:var(--text3)">(${m.year})</span></span>
      <span class="rdi-industry ind-${indClass(m.industry)}">${m.industry}</span>
    </div>`).join('');
  recDrop.classList.add('open');
});
document.addEventListener('click',e=>{if(!e.target.closest('.rec-search-wrap'))recDrop.classList.remove('open');});

function selectMovieForRec(id){
  const m=MOVIES.find(x=>x.id===id);if(!m)return;
  selectedMovieId=id;
  recInput.value=m.title;
  recDrop.classList.remove('open');
  const ic=indClass(m.industry);
  const card=document.getElementById('selectedMovieCard');
  card.style.display='flex';
  card.innerHTML=`
    <div class="smc-poster poster-${ic}">${m.emoji}</div>
    <div class="smc-info">
      <div class="smc-title">${m.title}</div>
      <div class="smc-meta">⭐ ${m.rating} · ${m.year} · ${m.language} · ${m.industry}</div>
      <div style="margin-top:6px">${streamBadgesHTML(m.streaming)}</div>
    </div>`;
  document.getElementById('getRecBtn').disabled=false;
}

function computeSimilarity(source, candidate){
  if(source.id===candidate.id)return -1;
  let score=0;
  // Genre overlap (max 40)
  const common=source.genre.filter(g=>candidate.genre.includes(g)).length;
  const union=[...new Set([...source.genre,...candidate.genre])].length;
  score += (common/union)*40;
  // Industry match (20)
  const sameInd=document.getElementById('prefSameIndustry').checked;
  if(source.industry===candidate.industry) score+= sameInd?20:10;
  // Language match (15)
  const sameLang=document.getElementById('prefSameLang').checked;
  if(source.language===candidate.language) score+= sameLang?15:8;
  // Era proximity (15)
  const sameEra=document.getElementById('prefSameEra').checked;
  const yearDiff=Math.abs(source.year-candidate.year);
  const eraScore=Math.max(0,15-(yearDiff*1.2));
  score+= sameEra ? eraScore : eraScore*0.5;
  // Rating proximity (10)
  const ratingDiff=Math.abs(source.rating-candidate.rating);
  score+=Math.max(0,10-ratingDiff*4);
  return Math.round(score);
}

document.getElementById('getRecBtn').addEventListener('click',()=>{
  if(!selectedMovieId)return;
  const source=MOVIES.find(m=>m.id===selectedMovieId);
  const scored=MOVIES
    .filter(m=>m.id!==source.id)
    .map(m=>({...m,score:computeSimilarity(source,m)}))
    .sort((a,b)=>b.score-a.score)
    .slice(0,8);

  const container=document.getElementById('recResults');
  container.innerHTML=scored.map((m,i)=>{
    const pct=Math.round((m.score/85)*100);
    return`<div class="rec-card" onclick="openModal(${m.id})" style="animation-delay:${i*0.07}s">
      <div class="rec-card-poster poster-${indClass(m.industry)}">${m.emoji}</div>
      <div class="rec-card-info">
        <div class="rec-card-title">${m.title} (${m.year})</div>
        <div class="rec-card-meta">${m.industry} · ${m.language} · ⭐${m.rating}</div>
        <div class="rec-card-streams">${streamBadgesHTML(m.streaming)}</div>
      </div>
      <div class="rec-card-score">
        <div class="rec-score-pct">${pct}%</div>
        <div class="rec-score-lbl">Match</div>
      </div>
    </div>`;
  }).join('');
  document.getElementById('recommend').scrollIntoView({behavior:'smooth'});
});

// Quick rec from movie card
function quickRec(id){
  selectMovieForRec(id);
  document.getElementById('getRecBtn').click();
  document.getElementById('recommend').scrollIntoView({behavior:'smooth'});
}

// ── ANALYTICS CHARTS ─────────────────────────────────────────
function buildChart(containerId, data, colorFn){
  const max=Math.max(...Object.values(data));
  const sorted=Object.entries(data).sort((a,b)=>b[1]-a[1]);
  document.getElementById(containerId).innerHTML=sorted.map(([k,v])=>`
    <div class="bc-row">
      <div class="bc-label"><span>${k}</span><span>${v}</span></div>
      <div class="bc-track"><div class="bc-fill" data-w="${(v/max)*100}" style="background:${colorFn(k)}"></div></div>
    </div>`).join('');
}

function countBy(key){ return MOVIES.reduce((acc,m)=>{const v=m[key];acc[v]=(acc[v]||0)+1;return acc;},{}); }
function countGenres(){return MOVIES.reduce((acc,m)=>{m.genre.forEach(g=>{acc[g]=(acc[g]||0)+1});return acc;},{});}
function countStreaming(){return MOVIES.reduce((acc,m)=>{m.streaming.forEach(s=>{acc[s]=(acc[s]||0)+1});return acc;},{});}

const IND_COLORS={Hollywood:'#f59e0b',Bollywood:'#f97316',Tollywood:'#10b981',Kollywood:'#ef4444',Mollywood:'#a855f7',Korean:'#ec4899',Kannada:'#06b6d4'};
const LANG_COLORS={English:'#06b6d4',Hindi:'#f97316',Telugu:'#10b981',Tamil:'#ef4444',Malayalam:'#a855f7',Korean:'#ec4899',Kannada:'#06b6d4'};
const GENRE_COLORS={Action:'#ef4444',Drama:'#a855f7',Thriller:'#f59e0b',Crime:'#ec4899','Sci-Fi':'#06b6d4',Romance:'#f43f5e',Comedy:'#84cc16',Biography:'#8b5cf6',Sport:'#f97316',Fantasy:'#06b6d4',Adventure:'#14b8a6',Horror:'#6366f1',Animation:'#22c55e',Mystery:'#d946ef',History:'#78716c',Music:'#f59e0b',Disaster:'#ef4444'};
const STREAM_COLORS={'Netflix':'#e50914','Amazon Prime':'#00a8e0','Disney+ Hotstar':'#0063e5','ZEE5':'#7b2fff','SonyLIV':'#e8541e','Apple TV+':'#555','HBO Max':'#5822a2','YouTube':'#ff0000','Hulu':'#1ce783'};

buildChart('industryChart',countBy('industry'),k=>IND_COLORS[k]||'#888');
buildChart('languageChart',countBy('language'),k=>LANG_COLORS[k]||'#888');
buildChart('genreChart',countGenres(),k=>GENRE_COLORS[k]||'#888');
buildChart('streamingChart',countStreaming(),k=>STREAM_COLORS[k]||'#888');

// --- SIMULATED USER ACTIVITY ---
function buildUserActivity() {
    const data = { 'Active Now': 12, 'Today': 84, 'This Week': 542, 'Total Users': 1205 };
    buildChart('userActivityChart', data, k => k.includes('Active') ? '#10b981' : '#a855f7');
}
buildUserActivity();

// Animate bars on scroll
const barIO=new IntersectionObserver(entries=>{
  if(entries[0].isIntersecting){
    document.querySelectorAll('.bc-fill').forEach(b=>{b.style.width=b.dataset.w+'%';});
    barIO.disconnect();
  }
},{threshold:.2});
const analyticsSection=document.getElementById('analytics');
if(analyticsSection) barIO.observe(analyticsSection);

// ── ACCORDION ────────────────────────────────────────────────
document.querySelectorAll('.acc-head').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const body=btn.nextElementSibling;
    const isOpen=body.classList.contains('open');
    document.querySelectorAll('.acc-body').forEach(b=>b.classList.remove('open'));
    document.querySelectorAll('.acc-head').forEach(h=>h.classList.remove('open'));
    if(!isOpen){body.classList.add('open');btn.classList.add('open');}
  });
});

// ── SCROLL ANIMATIONS ────────────────────────────────────────
const animIO=new IntersectionObserver(entries=>{
  entries.forEach(e=>{if(e.isIntersecting){e.target.style.opacity='1';e.target.style.transform='translateY(0)';animIO.unobserve(e.target);}});
},{threshold:.1});
document.querySelectorAll('.chart-card,.tool-badge,.acc-item').forEach(el=>{
  el.style.opacity='0';el.style.transform='translateY(20px)';el.style.transition='all .5s ease';
  animIO.observe(el);
});

// ── INIT ─────────────────────────────────────────────────────
applyFilters();
