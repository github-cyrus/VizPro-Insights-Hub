class Cart {
    constructor() {
        this.items = JSON.parse(localStorage.getItem('cart')) || [];
        this.total = 0;
        this.updateCartCount();
        this.calculateTotal();
    }

    addItem(item) {
        const existingItem = this.items.find(i => i.id === item.id);
        if (existingItem) {
            existingItem.quantity += 1;
        } else {
            this.items.push({ ...item, quantity: 1 });
        }
        this.saveCart();
        this.updateCartCount();
        this.calculateTotal();
        this.showNotification('Item added to cart');
    }

    removeItem(itemId) {
        this.items = this.items.filter(item => item.id !== itemId);
        this.saveCart();
        this.updateCartCount();
        this.calculateTotal();
    }

    saveCart() {
        localStorage.setItem('cart', JSON.stringify(this.items));
    }

    calculateTotal() {
        this.total = this.items.reduce((sum, item) => sum + (item.price * item.quantity), 0);
        document.querySelector('.cart-total').textContent = `₹${this.total}`;
    }

    updateCartCount() {
        const count = this.items.reduce((sum, item) => sum + item.quantity, 0);
        document.querySelector('.cart-count').textContent = count;
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'cart-notification';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => notification.remove(), 300);
            }, 2000);
        }, 100);
    }
}

// Initialize cart
const cart = new Cart();

// Add click handlers after DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add to cart button handler
    document.querySelectorAll('.add-to-cart').forEach(button => {
        button.addEventListener('click', (e) => {
            const card = e.target.closest('.course-card');
            const item = {
                id: card.dataset.courseId,
                name: card.querySelector('h4').textContent,
                price: parseInt(card.querySelector('.price span').textContent.replace('Rs-', '')),
                image: card.querySelector('.card-icon i').className
            };
            cart.addItem(item);
        });
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const cartTrigger = document.getElementById('cart-trigger');
    const cartSidebar = document.getElementById('cart-sidebar');
    const closeCart = document.getElementById('close-cart');
    const cartItems = document.getElementById('cart-items');
    const cartCount = document.querySelector('.cart-count');
    const totalAmount = document.querySelector('.total-amount');
    const checkoutBtn = document.getElementById('checkout-btn');

    // Cart state
    let cart = JSON.parse(localStorage.getItem('cart')) || [];

    // Update cart count
    function updateCartCount() {
        const total = cart.reduce((sum, item) => sum + item.quantity, 0);
        cartCount.textContent = total;
        localStorage.setItem('cart', JSON.stringify(cart));
    }

    // Calculate total price
    function calculateTotal() {
        const total = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
        totalAmount.textContent = `₹${total.toFixed(2)}`;
    }

    // Update cart display
    function updateCartDisplay() {
        cartItems.innerHTML = '';
        
        if (cart.length === 0) {
            cartItems.innerHTML = '<p style="text-align: center; color: #b0b0b0;">Your cart is empty</p>';
        } else {
            cart.forEach((item, index) => {
                cartItems.innerHTML += `
                    <div class="cart-item">
                        <div class="cart-item-details">
                            <h4 class="cart-item-title">${item.title}</h4>
                            <div class="cart-item-price">₹${item.price.toFixed(2)}</div>
                            <div class="cart-item-quantity">
                                <button onclick="window.updateQuantity(${index}, -1)" class="quantity-btn">-</button>
                                <span>${item.quantity}</span>
                                <button onclick="window.updateQuantity(${index}, 1)" class="quantity-btn">+</button>
                                <button onclick="window.removeItem(${index})" class="remove-item">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });
        }
        
        updateCartCount();
        calculateTotal();
    }

    // Add to cart function
    window.addToCart = function(title, price, image) {
        const existingItemIndex = cart.findIndex(item => item.title === title);
        
        if (existingItemIndex !== -1) {
            cart[existingItemIndex].quantity += 1;
        } else {
            cart.push({
                title,
                price: parseFloat(price),
                image,
                quantity: 1
            });
        }
        
        localStorage.setItem('cart', JSON.stringify(cart));
        updateCartDisplay();
        showNotification('Item added to cart!');
        cartSidebar.classList.add('active');
    };

    // Update quantity
    window.updateQuantity = function(index, change) {
        if (index >= 0 && index < cart.length) {
            cart[index].quantity += change;
            
            if (cart[index].quantity <= 0) {
                cart.splice(index, 1);
            }
            
            localStorage.setItem('cart', JSON.stringify(cart));
            updateCartDisplay();
        }
    };

    // Remove item
    window.removeItem = function(index) {
        if (index >= 0 && index < cart.length) {
            cart.splice(index, 1);
            localStorage.setItem('cart', JSON.stringify(cart));
            updateCartDisplay();
            showNotification('Item removed from cart!');
        }
    };

    // Show notification
    function showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'cart-notification';
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 2000);
    }

    // Cart event listeners
    cartTrigger.addEventListener('click', function(e) {
        e.preventDefault();
        cartSidebar.classList.add('active');
    });

    closeCart.addEventListener('click', function() {
        cartSidebar.classList.remove('active');
    });

    // Close cart when clicking outside
    document.addEventListener('click', function(e) {
        if (!cartSidebar.contains(e.target) && !cartTrigger.contains(e.target)) {
            cartSidebar.classList.remove('active');
        }
    });

    // Checkout button
    checkoutBtn.addEventListener('click', function() {
        if (cart.length === 0) {
            showNotification('Your cart is empty!');
            return;
        }
        showNotification('Proceeding to checkout...');
        // Add checkout logic here
        cartSidebar.classList.remove('active');
    });

    // Initialize cart display
    updateCartDisplay();
});
