//Here go our calls to the different algorithms...
var selectedFile="";
function runMethode(methodeName, params){ //RETURNS A PROMISE BECAUSE PYTHONSHELL MAY TAKE A LONG TIME...
    var path=require('path')
    var {PythonShell} =  require('python-shell');
    var opt ={
        args:[params] //parametres de la méthode
    }

    var promise=new Promise((resolve,reject)=>{
        PythonShell.run(path.join(__dirname,'scripts/'+methodeName),  opt,function  (err, results)  {
            if  (err)  {
                  alert(err);
                  throw err 
            };
            resolve(results[0])
        });
    })
    return promise;
}

function getDetails () {
    // Gets execution time, solution and path cost...
    var doc = document.getElementById('inputfile'); 
    var fileName=doc.files.item(0).name;
    selectedFile=fileName;
    //Here we open the file and get the details, given a file name...
    $("#execTime").text("Temps d'exécution: 0.3");
    $("#solution").text("Solution: 2,8,3,5");
    $("#pathCost").text("Cout: 54");
  }

function execute(){
      var s=$('form').serializeArray();
      var methodeName=s[0].value;
      if (selectedFile!=""){
        getParams(methodeName).then(v=>{
            v.fileName=selectedFile;
            runMethode(methodeName,JSON.stringify(v)).then((val)=>{
                var obj=JSON.parse(val);
                $("#execTime").text("Temps d'exécution: "+obj.execTime);
                $("#solution").text("Solution: Too Long!");
                $("#pathCost").text("Cout: "+obj.pathCost);
            }) 
          }) 
      } else {
          Swal.fire("Sélectionner un fichier du Benchmark SVP!")
      }
          
  }

async function getParams(methodeName){
    switch (methodeName){
        case 'simulated-annealing-tsp/test.py':{
            const {value:formValue}= await Swal.fire({
                    title: 'Parametres du récuit simulé',
                    html:
                      '<input id="swal-input1" class="swal2-input" placeholder="Temperature Finale" type="number">' +
                      '<input id="swal-input2" class="swal2-input" placeholder="Nombre d\'iteration" type="number">',
                    focusConfirm: false,
                    preConfirm: () => {
                        temperature=document.getElementById('swal-input1').value;
                        iterations=document.getElementById('swal-input2').value;
                        if(!temperature){
                            temperature=-1;
                        }
                        if(!iterations){
                            iterations=-1;
                        }
                        var par={
                            "temperature":temperature,
                            "iterations":iterations
                          }
                        return par
                    }
                })
                return formValue;
        }
        case 'GA-tsp/ga_call.py':{
            const {value:formValue}= await Swal.fire({
                    title: 'Parametres de l\'algo génétique',
                    html:
                      '<input id="swal-input1" class="swal2-input" placeholder="popSize" type="number">' +
                      '<input id="swal-input2" class="swal2-input" placeholder="eliteSize" type="number">'+
                      '<input id="swal-input3" class="swal2-input" placeholder="mutationRate" type="number">' +
                      '<input id="swal-input4" class="swal2-input" placeholder="generations" type="number">',
                    focusConfirm: false,
                    preConfirm: () => {
                        popSize=document.getElementById('swal-input1').value;
                        eliteSize=document.getElementById('swal-input2').value;
                        mutationRate=document.getElementById('swal-input3').value;
                        generations=document.getElementById('swal-input4').value;

                        if(!popSize){
                            popSize = 100;
                        }
                        if(!eliteSize){
                            eliteSize=30;
                        }
                        if(!mutationRate){
                            mutationRate=0.01;
                        }
                        if(!generations){
                            generations=500;
                        }
                        var par={
                            "popSize":popSize,
                            "eliteSize": eliteSize,
                            "mutationRate":mutationRate,
                            "generations":generations
                          }
                        return par
                    }
                })
                return formValue;
        }
        default:{
            var v={}
            return v;
        }
    }   
}

  