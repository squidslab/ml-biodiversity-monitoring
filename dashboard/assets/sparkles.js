/* --- ✨ LOGICA DELLE SCINTILLE (CLIENT-SIDE) --- */

document.addEventListener('click', function(e) {
    // Controlla se l'elemento cliccato ha la classe del nostro bottone
    if (e.target && e.target.classList.contains('magic-btn')) {
        createSparkles(e);
    }
});

function createSparkles(e) {
    const btn = e.target;
    
    // Quante scintille creare per click (es. 25-30)
    const numSparkles = 300 + Math.floor(Math.random() * 10);

    for (let i = 0; i < numSparkles; i++) {
        // Crea l'elemento DOM
        const sparkle = document.createElement('span');
        sparkle.classList.add('sparkle-element');
        
        // A volte creiamo una stellina, a volte una scintilla tonda
        if (Math.random() > 0.5) {
            sparkle.classList.add('star');
        }

        // Calcola una traiettoria casuale in ogni direzione
        // L'esplosione si espande da -150px a +150px dal centro
        const targetX = (Math.random() - 0.5) * 800 + "px";
        const targetY = (Math.random() - 0.5) * 900 + "px";
        
        // Imposta le variabili CSS casuali per l'animazione
        sparkle.style.setProperty('--target-x', targetX);
        sparkle.style.setProperty('--target-y', targetY);
        
        // Posiziona la scintilla esattamente dove l'utente ha cliccato
        // Usiamo pageX/pageY per precisione assoluta
        sparkle.style.left = (e.pageX) + "px";
        sparkle.style.top = (e.pageY) + "px";

        // Assegna una durata e un ritardo casuale all'animazione
        // (rende l'esplosione più naturale)
        const duration = (Math.random() * 1.5 + 1.4) + "s";
        const delay = (Math.random() * 0.1) + "s";
        sparkle.style.animation = `explodeAndFade ${duration} ${delay} cubic-bezier(0.25, 0.8, 0.25, 1) forwards`;

        // Aggiunge la scintilla al body della pagina
        document.body.appendChild(sparkle);

        // Rimuove l'elemento DOM dopo 1.5 secondi per non intasare la pagina
        setTimeout(() => {
            sparkle.remove();
        }, 1500);
    }
}