var requestURL = '/upload_file';

var request = new XMLHttpRequest();

var url = "/upload_file";

request.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
        var myArr = JSON.parse(this.responseText);
        myFunction(myArr);
    }
};
// request.open("GET", url, true);
// request.send();

function myFunction(arr) {
    console.log(arr);
    // document.getElementById("id01").innerHTML = out;
}


//   function populateHeader(jsonObj) {
//     var myH1 = document.createElement('h1');
//     myH1.textContent = jsonObj['Breed'];
//     header.appendChild(myH1);
  
//     var myPara = document.createElement('p');
//     myPara.textContent = 'Color: ' + jsonObj['Color'] + ' // Outcome Type: ' + jsonObj['Outcome Type'];
//     header.appendChild(myPara);
//   }

//   function showHeroes(jsonObj) {
//     var breed = jsonObj['members'];
        
//     for (var i = 0; i < heroes.length; i++) {
//       var myArticle = document.createElement('article');
//       var myH2 = document.createElement('h2');
//       var myPara1 = document.createElement('p');
//       var myPara2 = document.createElement('p');
//       var myPara3 = document.createElement('p');
//       var myList = document.createElement('ul');
  
//       myH2.textContent = heroes[i].name;
//       myPara1.textContent = 'Secret identity: ' + heroes[i].secretIdentity;
//       myPara2.textContent = 'Age: ' + heroes[i].age;
//       myPara3.textContent = 'Superpowers:';
          
//       var superPowers = heroes[i].powers;
//       for (var j = 0; j < superPowers.length; j++) {
//         var listItem = document.createElement('li');
//         listItem.textContent = superPowers[j];
//         myList.appendChild(listItem);
//       }
  
//       myArticle.appendChild(myH2);
//       myArticle.appendChild(myPara1);
//       myArticle.appendChild(myPara2);
//       myArticle.appendChild(myPara3);
//       myArticle.appendChild(myList);
  
//       section.appendChild(myArticle);
//     }
//   }