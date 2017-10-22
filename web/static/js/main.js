$(function(){
    
    $("#choose").change(function(){

        if(this.files.length){

            if($('#results').length){
                $('#results').empty();
            }

            var file = this.files[0];
            var canvas = document.getElementById('main');
            var ctx = canvas.getContext('2d');

            var image = new Image();

            var fr = new FileReader();

            fr.onload = function(evt){

                image.onload = function(){


                    //get the size of canvas;
                    var ch = canvas.height;
                    var cw = canvas.width;
                    
                    //get the size of image;
                    var image_height = image.naturalHeight;
                    var image_width = image.naturalWidth;

                    if(image_height >= image_width){
                        var mag = ch/image_height;
                    }else{
                        var mag = cw/image_width;
                    }

                    //resized image size;
                    var resized_height = image_height * mag;
                    var resized_width = image_width * mag;
                    ctx.fillStyle = "black";
                    ctx.fillRect(0, 0, cw, ch);
                    if(resized_height > resized_width){
                        var center = cw/2 - resized_width/2;
                        ctx.drawImage(image, center, 0, resized_width, resized_height);
                    }else{
                        var center = ch/2 - resized_height/2;
                        ctx.drawImage(image, 0, center, resized_width, resized_height);
                    }
                    

                    //var formData = new FormData($('#img_uplode')[0]);

                    var targetFile = $('input[name=img]');
                    var fd = new FormData();
                    var target = targetFile.eq(0);
                    fd.append('file', $(target).prop("files")[0]);

                    $.ajax({
                        url: '/api',
                        type: 'POST',
                        contentType: 'image/jpeg',
                        dataType: 'json',
                        data: fd,
                        contentType: false,
                        processData: false,
                    })
                    .success(function(data, statusText, jqXHR){

                        $('#faces').empty();
                        
                        if(data.results[0] != 0){

                            var dtype =  "data:image/jpg;base64,";
                            var src1 = data.results[0];
                            drawImage('main', src1);

                            var head = '<tr class="active"><th class="col-md-1">detected faces</th><th class="col-md-4">results</th></tr>';
                            
                            $('#faces').append(head);
                            
                            for(i = 0; i < data.results[1].length; i++){
                                var src = data.results[1][i];
                                var names = data.results[2];
                                var prob = data.results[3];
                                var element = '<tr><td><canvas id="input' + i.toString()+ '" style="border: 1px solid; margin: 12px 0 0 0" width="150" height="150"></canvas></td><td><table class="table"><tr class="active"><th>Candidates</th><th>probability</th>';
                                
                                for(j = 0; j < data.results[2].length; j++){
                                    element += '<tr><td>' + data.results[2][j] + '</td><td>' + (data.results[3][j] * 100).toFixed(2)+ '%<td></tr>';
                                }

                                $('#faces').append(element);

                                drawImage('input'+i.toString(), src);
                            }
                            //<tr style="border-bottom:1pt solid #cccccc">
                        }else{
                            $('<p></p>').text('No faces are detected.').appendTo('#faces');
                        }

                    })
                    .fail(function(jqXHR, statusText, errorThrown){
                        console.log(errorThrown);
                        console.log(statusText);
                        console.log(jqXHR);
                    });
                }
            image.src = evt.target.result;
            }
        fr.readAsDataURL(file);
        }
    });
});


drawImage = function(tag, img){
    
    var canvas = document.getElementById(tag);
    var ctx = canvas.getContext('2d');

    var image = new Image();

    image.onload = function(){

        //get the size of canvas;
        var ch = canvas.height;
        var cw = canvas.width;

        //get the size of image;
        var image_height = image.naturalHeight;
        var image_width = image.naturalWidth;

        if(image_height >= image_width){
            var mag = ch/image_height;
        }else{
            var mag = cw/image_width;
        }

        //resized image size;
        var resized_height = image_height * mag;
        var resized_width = image_width * mag;
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, cw, ch);
        if(resized_height > resized_width){
            var center = cw/2 - resized_width/2;
            ctx.drawImage(image, center, 0, resized_width, resized_height);
        }else{
            var center = ch/2 - resized_height/2;
            ctx.drawImage(image, 0, center, resized_width, resized_height);
        }
    }
    image.src = "data:image/jpg;base64," + img;
}


