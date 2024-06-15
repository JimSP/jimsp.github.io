fetch('descriptor.json')
    .then(response => response.json())
    .then(data => {
        const timeline = document.getElementById('timeline');

        data.forEach((values, key) => {
            for (let [dir, assets] of Object.entries(values)) {
                const item = document.createElement('div');
                item.className = 'timeline-item';
                item.innerHTML = '<label>' + dir + '</label></br>';
                timeline.appendChild(item);

                let onClickFunction;

                onClickFunction = () => {
                    console.log(item);
                    for (let [i, asset] of Object.entries(assets)) {
                        for (let [name, card] of Object.entries(asset)) {
                            const imgSrc = card;
                            const img = document.createElement('img');
                            img.src = imgSrc;
                            item.appendChild(img);
                        }
                    }
                    item.removeEventListener('click', onClickFunction);
                };
                item.addEventListener('click', onClickFunction);
            }
        });
    });