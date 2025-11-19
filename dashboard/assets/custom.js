// Robust scroll and tab position management for Streamlit
(function() {
    const SCROLL_KEY = 'racesense_scroll_position';
    const TAB_KEY = 'racesense_active_tab';
    const DRIVER_KEY = 'racesense_selected_driver';
    
    // Store complete state with multiple redundancy
    const storeState = () => {
        // Store scroll position and page height for intelligent restoration
        sessionStorage.setItem(SCROLL_KEY, window.scrollY.toString());
        sessionStorage.setItem('racesense_page_height', document.documentElement.scrollHeight.toString());
        
        // Store active tab with multiple methods for reliability
        const activeTab = document.querySelector('.stTabs [aria-selected="true"]');
        if (activeTab) {
            const tabIndex = Array.from(activeTab.parentElement.children).indexOf(activeTab);
            
            // Method 1: sessionStorage (backup)
            sessionStorage.setItem(TAB_KEY, tabIndex.toString());
            sessionStorage.setItem('racesense_active_tab', tabIndex.toString());
            
            // Method 2: URL query parameter (primary)
            const url = new URL(window.location);
            url.searchParams.set('tab', tabIndex);
            window.history.replaceState({}, '', url);
            
            // Method 3: localStorage (fallback for new tabs)
            try {
                localStorage.setItem('racesense_last_tab', tabIndex.toString());
            } catch(e) {}
        }
        
        // Store selected driver (from selectbox)
        const driverSelect = document.querySelector('[data-testid="stSelectbox"]');
        if (driverSelect) {
            const selectedOption = driverSelect.querySelector('[aria-selected="true"]');
            if (selectedOption) {
                sessionStorage.setItem(DRIVER_KEY, selectedOption.textContent);
            }
        }
    };
    
    // Restore scroll position with content-aware adjustment
    const restoreScrollPosition = () => {
        const scrollPosition = sessionStorage.getItem(SCROLL_KEY);
        const savedHeight = sessionStorage.getItem('racesense_page_height');
        
        if (scrollPosition !== null && scrollPosition !== '0') {
            requestAnimationFrame(() => {
                setTimeout(() => {
                    const currentHeight = document.documentElement.scrollHeight;
                    let targetScroll = parseInt(scrollPosition);
                    
                    // Adjust scroll if page height changed significantly
                    if (savedHeight) {
                        const heightDiff = currentHeight - parseInt(savedHeight);
                        // If page is much taller or shorter, adjust proportionally
                        if (Math.abs(heightDiff) > 200) {
                            const scrollRatio = targetScroll / parseInt(savedHeight);
                            targetScroll = Math.floor(scrollRatio * currentHeight);
                        }
                    }
                    
                    // Ensure we don't scroll past the bottom
                    const maxScroll = currentHeight - window.innerHeight;
                    targetScroll = Math.min(targetScroll, maxScroll);
                    
                    window.scrollTo({
                        top: Math.max(0, targetScroll),
                        behavior: 'instant'
                    });
                }, 250);
            });
        }
    };
    
    // Simplified restoration - only scroll position
    const restoreState = () => {
        restoreScrollPosition();
    };
    
    // Store state on all interactions
    const events = ['click', 'change', 'input', 'keyup'];
    events.forEach(event => {
        document.addEventListener(event, storeState, true);
    });
    
    // Store before navigation
    window.addEventListener('beforeunload', storeState);
    
    // Watch for Streamlit reruns - wait for content to stabilize
    let restoreTimeout;
    const observer = new MutationObserver((mutations) => {
        const hasChange = mutations.some(m => m.addedNodes.length > 0);
        if (hasChange) {
            // Clear previous timeout to wait for content to finish loading
            clearTimeout(restoreTimeout);
            // Wait for content to stabilize before restoring
            restoreTimeout = setTimeout(restoreState, 400);
        }
    });
    
    // Start observing
    const startObserving = () => {
        const target = document.querySelector('.main') || document.body;
        observer.observe(target, { childList: true, subtree: true });
    };
    
    // Initialize with multiple timing attempts for reliability
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            startObserving();
            restoreState();
        });
    } else {
        startObserving();
        restoreState();
    }
    
    // Multiple restore attempts at different timings to handle Streamlit's async rendering
    window.addEventListener('load', () => {
        setTimeout(restoreState, 100);
        setTimeout(restoreState, 300);
        setTimeout(restoreState, 600);
    });
    
    // Prevent scroll jump on specific elements and store state aggressively
    document.addEventListener('click', (e) => {
        if (e.target.closest('[data-testid="stCheckbox"]') || 
            e.target.closest('.stSelectbox') ||
            e.target.closest('[data-baseweb="tab"]') ||
            e.target.closest('[data-testid="stSelectbox"]')) {
            storeState();
        }
    }, true);
    
    // Store state when driver selection changes
    const observeDriverSelection = () => {
        const driverSelect = document.querySelector('[data-testid="stSelectbox"]');
        if (driverSelect) {
            driverSelect.addEventListener('change', storeState, true);
            driverSelect.addEventListener('click', storeState, true);
        }
    };
    
    // Observe for driver selectbox
    setTimeout(observeDriverSelection, 500);
    setInterval(observeDriverSelection, 2000);
    
})();

// Equalize heights of Quick Stats boxes and Weakest Segment cards
(function() {
    const equalizeBoxHeights = () => {
        // Equalize insight boxes
        const boxes = document.querySelectorAll('.insight-box');
        if (boxes.length > 0) {
            boxes.forEach(box => box.style.height = 'auto');
            let maxHeight = 0;
            boxes.forEach(box => {
                const height = box.offsetHeight;
                if (height > maxHeight) maxHeight = height;
            });
            if (maxHeight > 0) {
                boxes.forEach(box => box.style.height = maxHeight + 'px');
            }
        }
        
        // Equalize weakest segment cards
        const segmentCards = document.querySelectorAll('.weakest-segment-card');
        if (segmentCards.length > 0) {
            segmentCards.forEach(card => card.style.height = 'auto');
            let maxSegmentHeight = 0;
            segmentCards.forEach(card => {
                const height = card.offsetHeight;
                if (height > maxSegmentHeight) maxSegmentHeight = height;
            });
            if (maxSegmentHeight > 0) {
                segmentCards.forEach(card => card.style.height = maxSegmentHeight + 'px');
            }
        }
        
        // Equalize improvement recommendation cards
        const recCards = document.querySelectorAll('.improvement-rec-card');
        if (recCards.length > 0) {
            recCards.forEach(card => card.style.height = 'auto');
            let maxRecHeight = 0;
            recCards.forEach(card => {
                const height = card.offsetHeight;
                if (height > maxRecHeight) maxRecHeight = height;
            });
            if (maxRecHeight > 0) {
                recCards.forEach(card => card.style.height = maxRecHeight + 'px');
            }
        }
    };
    
    // Run on load and resize
    const runEqualize = () => {
        setTimeout(equalizeBoxHeights, 100);
        setTimeout(equalizeBoxHeights, 300);
        setTimeout(equalizeBoxHeights, 600);
    };
    
    // Initial run
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runEqualize);
    } else {
        runEqualize();
    }
    
    window.addEventListener('load', runEqualize);
    window.addEventListener('resize', equalizeBoxHeights);
    
    // Watch for Streamlit reruns
    const observer = new MutationObserver(() => {
        equalizeBoxHeights();
    });
    
    const startObserving = () => {
        const target = document.querySelector('.main') || document.body;
        observer.observe(target, { childList: true, subtree: true });
    };
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', startObserving);
    } else {
        startObserving();
    }
    
    // Watch for expander clicks to equalize heights when content becomes visible
    document.addEventListener('click', (e) => {
        if (e.target.closest('[data-testid="stExpander"]')) {
            setTimeout(equalizeBoxHeights, 100);
            setTimeout(equalizeBoxHeights, 300);
        }
    }, true);
    
})();
