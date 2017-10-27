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
                    

                    var targetFile = $('input[name=img]');
                    var fd = new FormData();
                    var target = targetFile.eq(0);
                    fd.append('file', $(target).prop("files")[0]);

                    $.ajax({
                        url: '/api',
                        type: 'POST',
                        contentType: 'image/jpeg',
                        //contentType: 'multipart/form-data'
                        dataType: 'json',
                        data: fd,
                        contentType: false,
                        processData: false,
                    })
                    .success(function(data, statusText, jqXHR){

                        $('#captions').empty();
                        $('#detail').empty();
                        
                        //var detail = '<p>' + String(data['jp']) + '</p>';
                        //$('detail').append(detail);

                        var languages = ['Japanese', 'English', 'Chinese'];
                        
                        var head = '<tr class="active"><th class="col-md-1">Language</th><th class="col-md-3">Detail</th></tr>';
                        $('#captions').append(head);
                        
                        var num_langs = Object.keys(data).length;
                        for(i = 0; i < num_langs; i++){
                            var caps = data[languages[i]];
                            var elements = '<tr><td class="lang">' + languages[i] + '</td><td><table class="table"><tr class="active"><th>No</th><th>Captions</th>';

                            num_caps = Object.keys(caps).length;
                            for(j = 0; j < num_caps; j++){
                                var cap = caps[j];
                                elements += '<tr><td>' + String(cap['No'] + 1) + '</td><td>' + cap['caption'] + '</td></tr>';
                            }
                            elements += '</table></td></tr>';
                            $('#captions').append(elements);
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


